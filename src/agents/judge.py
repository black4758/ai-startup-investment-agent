"""투자 판단 에이전트 (설계 문서 C.4, C.5) - 김영제 담당

총점, 과락, 판정은 코드(규칙)로 계산하고, LLM은 판정 사유만 작성한다.
- 재현성: 같은 점수면 항상 같은 총점과 판정
- 설계 B.1 "LLM 추론 + 규칙 기반"과 일치

retry_count 증가는 graph.py의 refine_and_retry,
evaluations 저장은 graph.py의 save_evaluation이 담당하므로 여기서는 다루지 않는다.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # 테스트 시 State 모듈 로딩 없이 타입 힌트만 사용
    from src.state import GraphState

# 설계 문서 C.1 평가표 (src/state.py 영문 키 기준)
WEIGHTS = {
    "tech_differentiation": 25,          # 기술 차별성 및 특허
    "founder_competence": 20,            # 창업자
    "market_position_and_barriers": 15,  # 시장 내 포지션 및 진입장벽
    "market_potential": 10,              # 시장성
    "track_record": 10,                  # 실적
    "business_model": 10,                # 수익모델
    "purchase_rationale": 10,            # 구매이유
}
CUTOFF_KEYS = ("tech_differentiation", "founder_competence")  # 설계 문서 C.5 과락 항목
MISSING_SCORE = 2   # 설계 문서 C.4 정보 부재 처리
PASS_LINE = 75
HOLD_LINE = 60

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "judge_prompt.txt"


def _to_score(value) -> float:
    """점수 값을 1~5 범위로 맞춘다. 값이 없거나 해석할 수 없으면 정보 부재(2점).

    에이전트가 준 소수 점수(예: 3.5)는 반올림하지 않고 그대로 쓴다.
    (파이썬 round는 .5를 짝수 쪽으로 반올림해 2.5→2, 3.5→4처럼 방향이 제각각이다)
    점수 규칙: 1~5점, 정보 부재는 2점. 0점은 1점으로 잘려 과락 처리되므로 쓰지 않는다.
    """
    if isinstance(value, dict):          # {"score": 4, "reason": ...} 형태도 허용
        value = value.get("score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(MISSING_SCORE)
    return max(1.0, min(5.0, score))


def merge_scores(state: "GraphState") -> Dict[str, float]:
    """세 에이전트의 점수를 합쳐 7개 항목 점수를 만든다. 빠진 항목은 2점."""
    merged = {}
    for field in ("tech_scores", "company_scores", "market_scores"):
        merged.update(state.get(field) or {})
    return {key: _to_score(merged.get(key)) for key in WEIGHTS}


def calculate_total(scores: Dict[str, float]) -> float:
    """총점 = Σ (점수 / 5) × 비중. 부동소수 오차로 60점이 59.99…가 되지 않도록 반올림."""
    total = sum(scores[key] / 5 * weight for key, weight in WEIGHTS.items())
    return round(total, 2)


def check_cutoff(scores: Dict[str, float]) -> bool:
    """과락 여부: 기술 차별성 및 특허 또는 창업자가 1점 이하."""
    return any(scores[key] <= 1 for key in CUTOFF_KEYS)


def decide(total: float, cutoff: bool) -> str:
    if cutoff:
        return "DROP"
    if total >= PASS_LINE:
        return "PASS"
    if total >= HOLD_LINE:
        return "HOLD"
    return "DROP"


def _build_prompt(state: "GraphState", total: float, decision: str, cutoff: bool) -> str:
    """.format()은 dict 문자열의 중괄호 때문에 깨지므로 replace로 치환한다."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    values = {
        "{startup}": state.get("current_startup", ""),
        "{tech_analysis}": state.get("tech_analysis", ""),
        "{tech_scores}": state.get("tech_scores", {}),
        "{company_analysis}": state.get("company_analysis", ""),
        "{company_scores}": state.get("company_scores", {}),
        "{market_analysis}": state.get("market_analysis", ""),
        "{market_scores}": state.get("market_scores", {}),
        "{missing_items}": state.get("missing_items", []) or "없음",
        "{total_score}": total,
        "{decision}": decision,
        "{cutoff}": "과락 발생" if cutoff else "과락 없음",
    }
    for placeholder, value in values.items():
        template = template.replace(placeholder, str(value))
    return template


def evaluate_judge(state: "GraphState") -> Dict[str, Any]:
    scores = merge_scores(state)
    total = calculate_total(scores)
    cutoff = check_cutoff(scores)
    decision = decide(total, cutoff)

    # 판정 사유는 LLM이 작성. 실패해도 판정 결과는 그대로 반환한다.
    try:
        from src.agents.tech import get_llm
        response = get_llm().invoke(_build_prompt(state, total, decision, cutoff))
        reason = getattr(response, "content", str(response))
    except Exception as e:
        reason = f"(판정 사유 생성 실패: {e})"

    print(f"[투자 판단] {state.get('current_startup', '')} | 점수 {scores}")
    print(f"[투자 판단] 총점 {total} / 판정 {decision} / {'과락 발생' if cutoff else '과락 없음'}")
    print(reason)

    return {"total_score": total, "decision": decision}
