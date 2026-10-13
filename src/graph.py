"""
[src/graph.py]
LangGraph 기반 Physical AI 스타트업 투자 평가 파이프라인 - Supervisor 패턴

구조 (허브-스포크):
    START -> extract_candidate -> supervisor
    supervisor --(add_conditional_edges, next_agent)--> tech | company | market | judge | extract_candidate | reporter
    tech / company / market / judge -> supervisor   (하위 에이전트 간 직접 연결 없음)
    reporter -> [quality_eval -> (FAIL: reporter 재작성 | PASS: END)]  (품질 평가 노드 연결 시)

Supervisor 판단 규칙 (규칙 기반 = 같은 State면 항상 같은 결정 -> 재현성):
    1. 미수집 관점이 있으면 -> 평가 가중치가 큰 관점부터 실행 (기업 40% > 기술 35% > 시장 25%)
    2. 3개 관점 수집 후 근거 충분도 검사 -> 부족한 관점만 재작업 (관점당 최대 1회)
       - 실행 실패 / 분석 없음 / 출처 2건 미만 / 항목 과반이 정보 부재(2점 이하)
    3. 근거가 충분하거나 재작업 상한 도달 -> judge
    4. 보완 재조사 없이 첫 HOLD -> 점수가 가장 낮은 관점만 보완 재조사 후 재판정
    5. 판정 확정 -> 남은 기업이 있으면 extract_candidate, 없으면 reporter

종료 보장 (3중 장치):
    - 관점별 실행 상한 MAX_AGENT_RUNS (agent_counts), 보고서 재작성 상한 MAX_REPORT_REVISIONS (report_revision_count)
    - 전체 Supervisor 결정 상한 MAX_SUPERVISOR_STEPS (step_count)
    - LangGraph recursion_limit (GRAPH_RECURSION_LIMIT)

관측성:
    - 결정과 사유 전체 이력은 외부 로그(outputs/decision_log_{trace_id}.jsonl)에 기록
    - State에는 최신 결정(next_agent)과 사유 1건(node_status["supervisor"])만 유지 -> State 무한 증식 방지
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

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

# 하위 에이전트(관점) 정의: 평가 가중치(%) 순으로 우선 수집, 점수 키는 state.py 스코어 타입에서 가져옴
PERSPECTIVES: Dict[str, Dict[str, Any]] = {
    "company": {"label": "기업", "weight": 40, "score_keys": list(CompanyScores.__annotations__)},
    "tech":    {"label": "기술", "weight": 35, "score_keys": list(TechScores.__annotations__)},
    "market":  {"label": "시장", "weight": 25, "score_keys": list(MarketScores.__annotations__)},
}

MIN_SOURCES = 2              # 관점당 최소 출처 수 (단일 출처 편중 방지 = 노션 '편향 통제')
WEAK_SCORE = 2.0             # 2점 이하 = 정보 부재 수준 (judge.py MISSING_SCORE와 동일 기준)
MAX_AGENT_RUNS = 2           # 관점당 최대 실행 횟수 (최초 1회 + 재작업 1회)
MAX_REPORT_REVISIONS = 3     # 품질 평가 탈락 시 보고서 재작성 상한 (state.py report_revision_count 주석 기준)
MAX_SUPERVISOR_STEPS = 60    # 전체 Supervisor 결정 상한 (기업당 최대 약 9회 x 5개사 + 여유)
GRAPH_RECURSION_LIMIT = 150  # LangGraph 노드 실행 상한 (마지막 안전망)

DECISION_LOG_DIR = Path("outputs")

# 팀원이 만든 품질 평가 노드 함수를 여기에 연결 (예: from src.agents.quality import evaluate_quality)
# None이면 reporter -> END로 연결되고, 함수가 지정되면 reporter -> quality_eval -> (재작성 | END)로 연결됨
QUALITY_EVAL_NODE: Optional[Callable[[GraphState], Dict[str, Any]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _perspective(state: GraphState, agent: str) -> Dict[str, Any]:
    """perspectives[agent]를 dict로 반환 (PerspectiveResult 객체와 dict 모두 허용)"""
    p = (state.get("perspectives") or {}).get(agent)
    if isinstance(p, BaseModel):
        return p.model_dump()
    return dict(p or {})


def _runs(state: GraphState, agent: str) -> int:
    return int((state.get("agent_counts") or {}).get(agent, 0))


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
    print(f"[{node}] -> {decision} | {reason}")
    try:
        DECISION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DECISION_LOG_DIR / f"decision_log_{trace_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:  # 로그 실패로 파이프라인이 멈추지 않도록
        print(f"[Decision Log] 기록 실패: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 근거 충분도 판정
# ─────────────────────────────────────────────────────────────────────────────

def _weak_items(state: GraphState, agent: str) -> List[str]:
    """정보 부재 수준(2점 이하 또는 점수 없음) 항목 목록"""
    scores = _perspective(state, agent).get("scores") or {}
    weak = []
    for key in PERSPECTIVES[agent]["score_keys"]:
        try:
            if float(scores.get(key)) <= WEAK_SCORE:
                weak.append(key)
        except (TypeError, ValueError):
            weak.append(key)
    return weak


def evidence_gaps(state: GraphState, agent: str) -> List[str]:
    """해당 관점의 근거 부족 사유 목록을 반환합니다. 빈 리스트면 근거 충분."""
    if (state.get("node_status") or {}).get(agent) == "FAIL":
        return ["에이전트 실행 실패"]

    p = _perspective(state, agent)
    gaps = []
    if not str(p.get("analysis") or "").strip():
        gaps.append("분석 없음")

    n_sources = len(set(p.get("sources") or []))
    if n_sources < MIN_SOURCES:
        gaps.append(f"출처 {n_sources}건 (기준 {MIN_SOURCES}건 이상)")

    # 항목 과반이 정보 부재일 때만 부족으로 판단 (1개 항목만 약한 경우는 judge의 2점 처리로 충분)
    weak = _weak_items(state, agent)
    if len(weak) * 2 > len(PERSPECTIVES[agent]["score_keys"]):
        gaps.append(f"정보 부재 항목 과반 {weak}")

    return gaps


def _score_ratio(state: GraphState, agent: str) -> float:
    """관점의 평균 점수 비율 (0~1). HOLD 시 가장 약한 관점을 고르는 데 사용."""
    scores = _perspective(state, agent).get("scores") or {}
    keys = PERSPECTIVES[agent]["score_keys"]
    values = []
    for key in keys:
        try:
            values.append(float(scores.get(key)))
        except (TypeError, ValueError):
            values.append(0.0)
    return sum(values) / (5.0 * len(keys)) if keys else 0.0


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
        if _runs(state, agent) == 0:
            spec = PERSPECTIVES[agent]
            return agent, f"{spec['label']} 관점 미수집 -> 실행 (평가 가중치 {spec['weight']}%)"

    # 2) 근거 부족 관점 재작업: 부족 사유가 많은 관점부터, 상한 미도달만
    gaps = {a: evidence_gaps(state, a) for a in by_weight}
    reworkable = [a for a in by_weight if gaps[a] and _runs(state, a) < MAX_AGENT_RUNS]
    if reworkable:
        target = max(reworkable, key=lambda a: len(gaps[a]))
        return target, f"{PERSPECTIVES[target]['label']} 근거 부족 {gaps[target]} -> 재작업"

    # 3) 보완 재조사 없이 첫 HOLD -> 점수가 가장 낮은 관점 보완 재조사
    if status.get("judge") == "HOLD_RETRY" and state.get("retry_count", 0) == 0:
        candidates = [a for a in by_weight if _runs(state, a) < MAX_AGENT_RUNS]
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
    """Supervisor 노드: 다음 노드를 결정해 next_agent에 기록. 실제 분기는 route_supervisor가 수행."""
    next_agent, reason = decide_next(state)
    log_decision(state, "supervisor", next_agent, reason)

    status = dict(state.get("node_status") or {})
    status["supervisor"] = f"{next_agent}: {reason}"   # 최신 사유 1건만 (전체 이력은 외부 로그)
    update: Dict[str, Any] = {
        "next_agent": next_agent,
        "node_status": status,
        "step_count": state.get("step_count", 0) + 1,
    }

    # 재작업이 시작되면 retry_count=1: company 에이전트는 보강 검색, judge는 다음 판정을 확정으로 처리
    if next_agent in PERSPECTIVES and _runs(state, next_agent) >= 1:
        update["retry_count"] = 1
    return update


def route_supervisor(state: GraphState) -> str:
    """add_conditional_edges 분기 함수: Supervisor가 기록한 next_agent로 이동"""
    return state.get("next_agent") or "reporter"


# ─────────────────────────────────────────────────────────────────────────────
# 4. 노드 (하위 에이전트 코드는 수정하지 않고 graph에서 감쌈)
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

    return {
        "candidates": candidates,
        "current_startup": current_startup,
        "perspectives": {},
        "agent_counts": {agent: 0 for agent in PERSPECTIVES},
        "retry_count": 0,
        "total_score": 0.0,
        "decision": "",
        "node_status": {},
        "last_error": None,
    }


def make_agent_node(agent: str, fn: Callable[[GraphState], Dict[str, Any]]) -> Callable[[GraphState], Dict[str, Any]]:
    """
    하위 에이전트 래퍼:
    - agent_counts에 실행 횟수 기록 (재작업 상한 판단)
    - 예외 발생 시 프로그램을 멈추지 않고 FAIL + last_error 기록 -> Supervisor가 재시도 여부 결정
    """
    def node(state: GraphState) -> Dict[str, Any]:
        counts = dict(state.get("agent_counts") or {})
        counts[agent] = counts.get(agent, 0) + 1
        status = dict(state.get("node_status") or {})

        try:
            result = dict(fn(state) or {})
        except Exception as e:
            status[agent] = "FAIL"
            error = f"[{agent}] {type(e).__name__}: {e}"
            print(f"[Fallback] {error}")
            return {"agent_counts": counts, "node_status": status, "last_error": error}

        status[agent] = "SUCCESS"
        result["agent_counts"] = counts
        result["node_status"] = status
        return result

    node.__name__ = agent
    return node


def make_judge_node(fn: Callable[[GraphState], Dict[str, Any]]) -> Callable[[GraphState], Dict[str, Any]]:
    """judge 래퍼: 판정 확정 여부를 node_status에 기록 (judge는 확정 시에만 evaluations를 반환)"""
    def judge(state: GraphState) -> Dict[str, Any]:
        result = dict(fn(state) or {})
        status = dict(state.get("node_status") or {})
        status["judge"] = "FINAL" if "evaluations" in result else "HOLD_RETRY"
        result["node_status"] = status
        return result

    return judge


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
        status = dict(state.get("node_status") or {})
        status.update(result.get("node_status") or {})  # 팀원 노드가 node_status를 반환해도 기존 기록 유지

        verdict, feedback = _verdict(result.get("eval_result", state.get("eval_result")))
        revisions = state.get("report_revision_count", 0)
        if verdict == "PASS":
            next_agent, reason = "end", "품질 평가 PASS -> 종료"
            status["quality_eval"] = "PASS"
        elif revisions < MAX_REPORT_REVISIONS:
            next_agent = "reporter"
            reason = f"품질 평가 {verdict or '판정 없음'} -> 보고서 재작성 {revisions + 1}회차 ({feedback[:80]})"
            result["report_revision_count"] = revisions + 1
            status["quality_eval"] = "FAIL"
        else:
            next_agent = "end"
            reason = f"품질 평가 미달이나 재작성 상한({MAX_REPORT_REVISIONS}회) 도달 -> 경고와 함께 종료"
            status["quality_eval"] = "FAIL_ACCEPTED"

        log_decision(state, "quality_eval", next_agent, reason)
        result["next_agent"] = next_agent
        result["node_status"] = status
        return result

    return quality_eval


def route_after_quality(state: GraphState) -> str:
    return "reporter" if state.get("next_agent") == "reporter" else "end"


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
    workflow.add_node("reporter", fns["reporter"])

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

    initial_state: GraphState = {
        "candidates": ["트위니", "뉴빌리티", "서울로보틱스", "마스오토", "엑스와이지"],
        "current_startup": "",
        "perspectives": {},
        "total_score": 0.0,
        "decision": "",
        "evaluations": [],
        "report": "",
        "eval_result": None,
        "next_agent": "",
        "agent_counts": {agent: 0 for agent in PERSPECTIVES},
        "report_revision_count": 0,
        "trace_id": str(uuid.uuid4())[:8],
        "step_count": 0,
        "retry_count": 0,
        "node_status": {},
        "last_error": None,
    }

    print("🚀 [Graph] Physical AI 스타트업 투자 평가 그래프 실행을 시작합니다...")
    result = graph.invoke(initial_state)
    print("\n🎉 [Graph] 전체 파이프라인 실행이 완료되었습니다!")
    print(f"총 평가된 기업 수: {len(result.get('evaluations', []))}개사")
    print(f"최종 보고서 길이: {len(result.get('report', ''))}자")
