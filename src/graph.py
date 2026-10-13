"""
[src/graph.py]
LangGraph 기반 Physical AI 스타트업 투자 평가 파이프라인 - Supervisor 패턴

구조 (허브-스포크):
    START -> extract_candidate -> supervisor
    supervisor --(add_conditional_edges)--> tech | company | market | judge | extract_candidate | reporter
    tech / company / market / judge -> supervisor   (하위 에이전트 간 직접 연결 없음)
    reporter -> [quality_eval -> (FAIL: reporter 재작성 | PASS: END)]  (품질 평가 노드 연결 시)

Supervisor 판단 규칙 (규칙 기반 = 같은 State면 항상 같은 결정 -> 재현성):
    1. 미수집 관점이 있으면 -> 평가 가중치가 큰 관점부터 실행 (기업 40% > 기술 35% > 시장 25%)
    2. 3개 관점 수집 후 근거 충분도 검사 -> 부족한 관점만 재작업 (관점당 최대 1회)
       - 실행 실패 / 분석 텍스트 없음 / 출처 2건 미만 / 정보 부재(2점 이하) 항목 존재
    3. 근거가 충분하거나 재작업 상한 도달 -> judge
    4. judge가 첫 HOLD -> 점수가 가장 낮은 관점만 보완 재조사 후 재판정 (기존 전체 재평가 대체)
    5. 판정 확정 -> 남은 기업이 있으면 extract_candidate, 없으면 reporter

종료 보장 (3중 장치):
    - 관점별 실행 상한 MAX_AGENT_RUNS, 보고서 작성 상한 MAX_REPORT_RUNS
    - 전체 Supervisor 결정 상한 MAX_SUPERVISOR_STEPS (step_count)
    - LangGraph recursion_limit (GRAPH_RECURSION_LIMIT)

관측성:
    - Supervisor 결정과 사유를 외부 로그(outputs/decision_log_{trace_id}.jsonl)에 기록 (State에는 최신 1건만)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from src.state import GraphState, TechScores, CompanyScores, MarketScores
from src.agents.tech import evaluate_technology
from src.agents.company import evaluate_company
from src.agents.market import evaluate_market
from src.agents.judge import evaluate_judge
from src.agents.reporter import generate_report

# .env 환경변수 로드
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 0. 설정값
# ─────────────────────────────────────────────────────────────────────────────

# 하위 에이전트(관점) 정의: 평가 가중치(%) 순으로 우선 실행, 점수 키는 state.py 스코어 타입에서 가져옴
PERSPECTIVES: Dict[str, Dict[str, Any]] = {
    "company": {"label": "기업", "weight": 40, "scores_field": "company_scores",
                "analysis_field": "company_analysis", "score_keys": list(CompanyScores.__annotations__)},
    "tech":    {"label": "기술", "weight": 35, "scores_field": "tech_scores",
                "analysis_field": "tech_analysis", "score_keys": list(TechScores.__annotations__)},
    "market":  {"label": "시장", "weight": 25, "scores_field": "market_scores",
                "analysis_field": "market_analysis", "score_keys": list(MarketScores.__annotations__)},
}

MIN_SOURCES = 2              # 관점당 최소 출처 수 (단일 출처 편중 방지 = 노션 '편향 통제')
WEAK_SCORE = 2.0             # 2점 이하 = 정보 부재 수준 (judge.py MISSING_SCORE와 동일 기준)
MAX_AGENT_RUNS = 2           # 관점당 최대 실행 횟수 (최초 1회 + 재작업 1회)
MAX_REPORT_RUNS = 2          # 보고서 최대 작성 횟수 (최초 1회 + 품질 미달 재작성 1회)
MAX_SUPERVISOR_STEPS = 60    # 전체 Supervisor 결정 상한 (기업당 최대 약 9회 x 5개사 + 여유)
GRAPH_RECURSION_LIMIT = 150  # LangGraph 노드 실행 상한 (마지막 안전망)

DECISION_LOG_DIR = Path("outputs")

# 팀원이 만든 품질 평가 노드 함수를 여기에 연결 (예: from src.agents.quality import evaluate_quality)
# None이면 reporter -> END로 연결되고, 함수가 지정되면 reporter -> quality_eval -> (재작성 | END)로 연결됨
QUALITY_EVAL_NODE: Optional[Callable[[GraphState], Dict[str, Any]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. node_status 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
# [임시 - State 확정 후 정리 대상]
# 현재 state.py에는 next_agent, 관점별 실행 횟수/출처 수, 보고서 작성 횟수 칸이 없어
# 기존 node_status(Dict[str, str])에 "관점:항목" 키로 기록합니다. 사용하는 키:
#   "{agent}"           : SUCCESS / FAIL            (재개/복구 판단)
#   "{agent}:runs"      : 현재 기업에서 실행 횟수    (재작업 상한)
#   "{agent}:sources"   : 마지막 실행의 출처 수      (근거 충분도)
#   "judge"             : FINAL / HOLD_RETRY        (판정 확정 여부)
#   "supervisor:next"   : Supervisor가 정한 다음 노드
#   "supervisor:reason" : 최신 결정 사유 (전체 이력은 외부 로그)
#   "offset:sources" / "offset:missing" : 현재 기업 시작 시점의 누적 리스트 길이
#   "reporter:runs" / "quality:next"    : 보고서 작성 횟수 / 품질 평가 후 다음 노드

def _status(state: GraphState) -> Dict[str, str]:
    return dict(state.get("node_status") or {})


def _get_int(status: Dict[str, str], key: str) -> int:
    try:
        return int(status.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _current_slice(state: GraphState, field: str) -> List[Any]:
    """
    [임시 - State 확정 후 정리 대상]
    sources / missing_items는 operator.add 누적 필드라 이전 기업 항목이 섞입니다.
    기업 시작 시점의 길이(offset)를 기억해 두고, 그 이후 항목만 현재 기업 것으로 봅니다.
    """
    offset = _get_int(state.get("node_status") or {}, f"offset:{field.split('_')[0]}")
    return list(state.get(field) or [])[offset:]


def log_decision(state: GraphState, node: str, decision: str, reason: str) -> None:
    """결정과 사유를 콘솔과 외부 JSONL 로그에 기록 (노션 State 샘플: 결정 로그는 State 밖 관측성 계층으로)"""
    trace_id = state.get("trace_id") or "no-trace"
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "trace_id": trace_id,
        "step": state.get("step_count", 0),
        "node": node,
        "startup": state.get("current_startup", ""),
        "decision": decision,
        "reason": reason,
    }
    print(f"[{node.capitalize()}] -> {decision} | {reason}")
    try:
        DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DECISION_LOG_DIR / f"decision_log_{trace_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:  # 로그 실패로 파이프라인이 멈추지 않도록
        print(f"[Decision Log] 기록 실패: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 근거 충분도 판정
# ─────────────────────────────────────────────────────────────────────────────

def evidence_gaps(state: GraphState, agent: str) -> List[str]:
    """해당 관점의 근거 부족 사유 목록을 반환합니다. 빈 리스트면 근거 충분."""
    spec = PERSPECTIVES[agent]
    status = state.get("node_status") or {}
    gaps = []

    if status.get(agent) == "FAIL":
        return ["에이전트 실행 실패"]

    if not str(state.get(spec["analysis_field"]) or "").strip():
        gaps.append("분석 텍스트 없음")

    n_sources = _get_int(status, f"{agent}:sources")
    if n_sources < MIN_SOURCES:
        gaps.append(f"출처 {n_sources}건 (기준 {MIN_SOURCES}건 이상)")

    scores = state.get(spec["scores_field"]) or {}
    weak = []
    for key in spec["score_keys"]:
        try:
            if float(scores.get(key)) <= WEAK_SCORE:
                weak.append(key)
        except (TypeError, ValueError):
            weak.append(key)
    if weak:
        gaps.append(f"정보 부재 수준 항목 {weak}")

    return gaps


def _score_ratio(state: GraphState, agent: str) -> float:
    """관점의 평균 점수 비율 (0~1). HOLD 시 가장 약한 관점을 고르는 데 사용."""
    spec = PERSPECTIVES[agent]
    scores = state.get(spec["scores_field"]) or {}
    values = []
    for key in spec["score_keys"]:
        try:
            values.append(float(scores.get(key)))
        except (TypeError, ValueError):
            values.append(0.0)
    return sum(values) / (5.0 * len(values)) if values else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Supervisor
# ─────────────────────────────────────────────────────────────────────────────

def decide_next(state: GraphState) -> Tuple[str, str]:
    """
    현재 State만 보고 다음 노드와 사유를 결정하는 순수 함수 (LLM 미사용 -> 재현성 보장).
    반환: (다음 노드 이름, 결정 사유)
    """
    status = state.get("node_status") or {}
    startup = state.get("current_startup", "")
    by_weight = sorted(PERSPECTIVES, key=lambda a: -PERSPECTIVES[a]["weight"])

    # [종료 보장] 전체 결정 상한
    if state.get("step_count", 0) >= MAX_SUPERVISOR_STEPS:
        return "reporter", f"Supervisor 결정 상한({MAX_SUPERVISOR_STEPS}회) 도달 -> 현재까지 결과로 보고서 작성"

    # 판정 확정 -> 다음 기업 또는 보고서
    if status.get("judge") == "FINAL":
        decision = state.get("decision", "")
        remaining = len(state.get("candidates") or [])
        if remaining:
            return "extract_candidate", f"'{startup}' 판정 확정({decision}) -> 다음 기업 (남은 {remaining}개사)"
        return "reporter", f"'{startup}' 판정 확정({decision}), 전 기업 평가 완료 -> 보고서 작성"

    # 1) 미수집 관점: 가중치가 큰 관점부터
    for agent in by_weight:
        if _get_int(status, f"{agent}:runs") == 0:
            spec = PERSPECTIVES[agent]
            return agent, f"{spec['label']} 관점 미수집 -> 실행 (평가 가중치 {spec['weight']}%)"

    # 2) 근거 부족 관점 재작업: 부족 사유가 많은 관점부터, 상한 미도달만
    gaps = {a: evidence_gaps(state, a) for a in by_weight}
    reworkable = [a for a in by_weight if gaps[a] and _get_int(status, f"{a}:runs") < MAX_AGENT_RUNS]
    if reworkable:
        target = max(reworkable, key=lambda a: len(gaps[a]))
        return target, f"{PERSPECTIVES[target]['label']} 근거 부족 {gaps[target]} -> 재작업"

    # 3) 첫 HOLD 판정 -> 점수가 가장 낮은 관점 보완 재조사
    if status.get("judge") == "HOLD_RETRY" and state.get("retry_count", 0) == 0:
        candidates = [a for a in by_weight if _get_int(status, f"{a}:runs") < MAX_AGENT_RUNS]
        if candidates:
            target = min(candidates, key=lambda a: _score_ratio(state, a))
            return target, (f"HOLD 판정({state.get('total_score')}점) -> 가장 약한 "
                            f"{PERSPECTIVES[target]['label']} 관점(점수 비율 {_score_ratio(state, target):.0%}) 보완 재조사")
        return "judge", "HOLD 판정이나 모든 관점 재작업 상한 도달 -> 재판정으로 확정"

    # 4) 근거 충분 (또는 재작업 상한 도달) -> 투자 판단
    leftover = {PERSPECTIVES[a]["label"]: g for a, g in gaps.items() if g}
    if leftover:
        return "judge", f"재작업 상한 도달, 남은 부족 {leftover}은 정보 부재로 처리 -> 투자 판단"
    return "judge", "3개 관점 근거 충분 -> 투자 판단"


def supervisor(state: GraphState) -> Dict[str, Any]:
    """Supervisor 노드: 다음 노드를 결정하고 사유를 기록합니다. 실제 분기는 route_supervisor가 수행."""
    next_node, reason = decide_next(state)
    log_decision(state, "supervisor", next_node, reason)

    status = _status(state)
    status["supervisor:next"] = next_node
    status["supervisor:reason"] = reason
    update: Dict[str, Any] = {"node_status": status, "step_count": state.get("step_count", 0) + 1}

    # 재작업이 시작되면 retry_count=1: company 에이전트는 보강 검색, judge는 다음 판정을 확정으로 처리
    is_rework = next_node in PERSPECTIVES and _get_int(status, f"{next_node}:runs") >= 1
    if is_rework:
        update["retry_count"] = 1
    return update


def route_supervisor(state: GraphState) -> str:
    """add_conditional_edges 분기 함수: Supervisor가 State에 기록한 다음 노드로 이동"""
    return (state.get("node_status") or {}).get("supervisor:next", "reporter")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 노드 래퍼 (하위 에이전트 코드는 수정하지 않고 graph에서 감쌈)
# ─────────────────────────────────────────────────────────────────────────────

def extract_candidate(state: GraphState) -> Dict[str, Any]:
    """
    후보 기업 추출 노드:
    candidates 큐의 맨 앞 기업을 current_startup으로 지정하고 기업 단위 필드를 초기화합니다.
    """
    candidates = list(state.get("candidates") or [])
    if not candidates:
        raise ValueError("후보 기업 큐(candidates)가 비어 있습니다.")

    current_startup = candidates.pop(0)
    print(f"\n{'='*60}\n[Pipeline] 다음 평가 기업 시작: '{current_startup}' (남은 대기 큐: {len(candidates)}개사)\n{'='*60}")

    # 기업이 바뀌면 node_status도 새로 시작 (누적 리스트의 현재 길이만 offset으로 기억)
    status = {
        "offset:sources": str(len(state.get("sources") or [])),
        "offset:missing": str(len(state.get("missing_items") or [])),
    }
    return {
        "candidates": candidates,
        "current_startup": current_startup,
        "retry_count": 0,
        "tech_analysis": "",
        "tech_scores": {},
        "company_analysis": "",
        "company_scores": {},
        "market_analysis": "",
        "market_scores": {},
        "total_score": 0.0,
        "decision": "",
        "node_status": status,
        "last_error": None,
    }


def make_agent_node(agent: str, fn: Callable[[GraphState], Dict[str, Any]]) -> Callable[[GraphState], Dict[str, Any]]:
    """
    하위 에이전트 래퍼:
    - 실행 횟수/출처 수/성공 여부를 node_status에 기록 (Supervisor 판단 근거)
    - 예외 발생 시 프로그램을 멈추지 않고 FAIL + last_error 기록 -> Supervisor가 재시도 여부 결정
    - 재작업 시 이미 쌓인 출처/정보 부재 항목은 다시 누적하지 않음 (지속성 비용 관리)
    """
    def node(state: GraphState) -> Dict[str, Any]:
        status = _status(state)
        status[f"{agent}:runs"] = str(_get_int(status, f"{agent}:runs") + 1)

        try:
            result = dict(fn(state) or {})
        except Exception as e:
            status[agent] = "FAIL"
            status[f"{agent}:sources"] = "0"
            error = f"[{agent}] {type(e).__name__}: {e}"
            print(f"[Fallback] {error}")
            return {"node_status": status, "last_error": error}

        sources = list(dict.fromkeys(result.get("sources") or []))
        status[agent] = "SUCCESS"
        status[f"{agent}:sources"] = str(len(sources))

        existing_sources = set(_current_slice(state, "sources"))
        existing_missing = set(_current_slice(state, "missing_items"))
        result["sources"] = [s for s in sources if s not in existing_sources]
        result["missing_items"] = [m for m in (result.get("missing_items") or []) if m not in existing_missing]
        result["node_status"] = status
        return result

    node.__name__ = agent
    return node


def make_judge_node(fn: Callable[[GraphState], Dict[str, Any]]) -> Callable[[GraphState], Dict[str, Any]]:
    """judge 래퍼: 현재 기업의 정보 부재 항목만 전달하고, 판정 확정 여부를 node_status에 기록"""
    def judge(state: GraphState) -> Dict[str, Any]:
        view = dict(state)
        view["missing_items"] = _current_slice(state, "missing_items")
        result = dict(fn(view) or {})

        status = _status(state)
        # judge는 판정 확정 시에만 evaluations를 반환 (첫 HOLD는 재평가 대상)
        status["judge"] = "FINAL" if "evaluations" in result else "HOLD_RETRY"
        status["judge:runs"] = str(_get_int(status, "judge:runs") + 1)
        result["node_status"] = status
        return result

    return judge


def make_reporter_node(fn: Callable[[GraphState], Dict[str, Any]]) -> Callable[[GraphState], Dict[str, Any]]:
    """reporter 래퍼: 보고서 작성 횟수를 기록 (품질 평가 재작성 상한용)"""
    def reporter(state: GraphState) -> Dict[str, Any]:
        result = dict(fn(state) or {})
        status = _status(state)
        status["reporter:runs"] = str(_get_int(status, "reporter:runs") + 1)
        result["node_status"] = status
        return result

    return reporter


def _verdict(eval_result: Any) -> Tuple[str, str]:
    """eval_result(QualityEvalResult 또는 dict)에서 (verdict, feedback) 추출"""
    if eval_result is None:
        return "", ""
    if isinstance(eval_result, dict):
        return str(eval_result.get("verdict", "")).upper(), str(eval_result.get("feedback", ""))
    return str(getattr(eval_result, "verdict", "")).upper(), str(getattr(eval_result, "feedback", ""))


def make_quality_node(fn: Callable[[GraphState], Dict[str, Any]]) -> Callable[[GraphState], Dict[str, Any]]:
    """품질 평가 래퍼: 팀원 노드를 실행하고, 통과/재작성/상한 종료를 결정해 사유를 기록"""
    def quality_eval(state: GraphState) -> Dict[str, Any]:
        result = dict(fn(state) or {})
        status = _status(state)
        status.update(result.get("node_status") or {})  # 팀원 노드가 node_status를 반환해도 기존 기록 유지

        verdict, feedback = _verdict(result.get("eval_result", state.get("eval_result")))
        runs = _get_int(status, "reporter:runs")
        if verdict == "PASS":
            next_node, reason = "end", "품질 평가 PASS -> 종료"
        elif runs < MAX_REPORT_RUNS:
            next_node, reason = "reporter", f"품질 평가 {verdict or '판정 없음'} -> 보고서 재작성 ({feedback[:80]})"
        else:
            next_node, reason = "end", f"품질 평가 미달이나 재작성 상한({MAX_REPORT_RUNS}회) 도달 -> 경고와 함께 종료"
            status["quality"] = "FAIL_ACCEPTED"

        log_decision(state, "quality_eval", next_node, reason)
        status["quality:next"] = next_node
        result["node_status"] = status
        return result

    return quality_eval


def route_after_quality(state: GraphState) -> str:
    return (state.get("node_status") or {}).get("quality:next", "end")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 그래프 구성 및 컴파일
# ─────────────────────────────────────────────────────────────────────────────

def create_investment_graph(
    agents: Optional[Dict[str, Callable[[GraphState], Dict[str, Any]]]] = None,
    quality_eval: Optional[Callable[[GraphState], Dict[str, Any]]] = QUALITY_EVAL_NODE,
):
    """
    Supervisor 패턴 StateGraph 생성 및 컴파일.
    agents로 노드 함수를 바꿔 끼울 수 있음 (API 호출 없는 구조 테스트용).
    """
    fns = {
        "tech": evaluate_technology,
        "company": evaluate_company,
        "market": evaluate_market,
        "judge": evaluate_judge,
        "reporter": generate_report,
        **(agents or {}),
    }

    workflow = StateGraph(GraphState)

    # 노드 등록
    workflow.add_node("extract_candidate", extract_candidate)
    workflow.add_node("supervisor", supervisor)
    for agent in PERSPECTIVES:
        workflow.add_node(agent, make_agent_node(agent, fns[agent]))
    workflow.add_node("judge", make_judge_node(fns["judge"]))
    workflow.add_node("reporter", make_reporter_node(fns["reporter"]))

    # 시작 -> 기업 추출 -> Supervisor
    workflow.add_edge(START, "extract_candidate")
    workflow.add_edge("extract_candidate", "supervisor")

    # Supervisor -> State 기반 동적 분기
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "tech": "tech",
            "company": "company",
            "market": "market",
            "judge": "judge",
            "extract_candidate": "extract_candidate",
            "reporter": "reporter",
        },
    )

    # 하위 에이전트와 judge는 항상 Supervisor로 복귀 (에이전트 간 직접 통신 금지)
    for agent in [*PERSPECTIVES, "judge"]:
        workflow.add_edge(agent, "supervisor")

    # 보고서 -> 품질 평가 Loop
    if quality_eval is not None:
        workflow.add_node("quality_eval", make_quality_node(quality_eval))
        workflow.add_edge("reporter", "quality_eval")
        workflow.add_conditional_edges(
            "quality_eval",
            route_after_quality,
            {"reporter": "reporter", "end": END},
        )
    else:
        # 품질 평가 노드 연결 전: QUALITY_EVAL_NODE에 팀원 함수를 지정하면 위 Loop로 자동 전환
        workflow.add_edge("reporter", END)

    return workflow.compile().with_config(recursion_limit=GRAPH_RECURSION_LIMIT)


# 전역 실행 객체
graph = create_investment_graph()


# ─────────────────────────────────────────────────────────────────────────────
# 6. 직접 실행 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    initial_candidates = ["트위니", "뉴빌리티", "서울로보틱스", "마스오토", "엑스와이지"]
    initial_state: GraphState = {
        "candidates": initial_candidates,
        "current_startup": "",
        "tech_analysis": "",
        "tech_scores": {},
        "company_analysis": "",
        "company_scores": {},
        "market_analysis": "",
        "market_scores": {},
        "missing_items": [],
        "total_score": 0.0,
        "decision": "",
        "retry_count": 0,
        "evaluations": [],
        "sources": [],
        "report": "",
        "eval_result": None,
        "trace_id": str(uuid.uuid4())[:8],
        "step_count": 0,
        "node_status": {},
        "last_error": None,
    }

    print("🚀 [Graph] Physical AI 스타트업 투자 평가 그래프 실행을 시작합니다...")
    result = graph.invoke(initial_state)
    print("\n🎉 [Graph] 전체 파이프라인 실행이 완료되었습니다!")
    print(f"총 평가된 기업 수: {len(result.get('evaluations', []))}개사")
    print(f"최종 보고서 길이: {len(result.get('report', ''))}자")
