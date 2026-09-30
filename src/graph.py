"""
[src/graph.py]
LangGraph 기반 Physical AI 스타트업 투자 평가 파이프라인 (PDF D.2 Graph 흐름 구현)

흐름:
1. START -> extract_candidate (큐에서 1개사 추출 및 필드 초기화)
2. tech -> company -> market (3대 에이전트 순차 평가)
3. judge (가중 채점 및 PASS, HOLD, DROP 판정)
4. route_after_judge (조건부 분기):
   - HOLD이고 retry_count == 0 -> refine_and_retry (보완 재검색 1회) -> tech 재진입
   - PASS, DROP 또는 retry_count >= 1 -> save_evaluation (평가 결과 누적)
5. route_after_save (조건부 분기):
   - 남은 기업 존재 (len(candidates) > 0) -> extract_candidate (Loop!)
   - 5개사 전원 검토 완료 (len(candidates) == 0) -> reporter (보고서 생성)
6. reporter -> END
"""

import os
from typing import Any, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from src.state import GraphState
from src.agents.tech import evaluate_technology
from src.agents.company import evaluate_company
from src.agents.market import evaluate_market
from src.agents.judge import evaluate_judge
from src.agents.reporter import generate_report

# .env 환경변수 로드
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 1. 그래프 제어 노드 정의
# ─────────────────────────────────────────────────────────────────────────────

def extract_candidate(state: GraphState) -> Dict[str, Any]:
    """
    후보 기업 추출 노드:
    candidates 큐의 맨 앞 기업을 꺼내어 current_startup으로 지정하고,
    단일 기업 평가 필드를 초기화합니다.
    """
    candidates = list(state.get("candidates") or [])
    if not candidates:
        raise ValueError("후보 기업 큐(candidates)가 비어 있습니다.")

    current_startup = candidates.pop(0)
    print(f"\n{'='*60}\n[Pipeline] 다음 평가 기업 시작: '{current_startup}' (남은 대기 큐: {len(candidates)}개사)\n{'='*60}")

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
    }


def refine_and_retry(state: GraphState) -> Dict[str, Any]:
    """
    재검색 및 보완 노드:
    HOLD 판정 시 정보 부재 항목을 재검색할 수 있도록 retry_count를 증가시킵니다. (최대 1회)
    """
    startup = state.get("current_startup", "")
    retry_count = state.get("retry_count", 0) + 1
    missing = state.get("missing_items", [])
    print(f"[Retry Loop] '{startup}' HOLD 판정으로 보완 재평가 루프 진입 (재시도 회차: {retry_count}, 정보 부재: {missing})")

    return {
        "retry_count": retry_count,
    }


def save_evaluation(state: GraphState) -> Dict[str, Any]:
    """
    현재 기업의 평가 결과를 evaluations 누적 리스트에 기록합니다.
    (operator.add 리듀서에 의해 자동 append)
    """
    startup = state.get("current_startup", "")
    total = state.get("total_score", 0.0)
    decision = state.get("decision", "")

    eval_record = {
        "startup": startup,
        "total_score": total,
        "decision": decision,
        "tech_scores": state.get("tech_scores") or {},
        "company_scores": state.get("company_scores") or {},
        "market_scores": state.get("market_scores") or {},
        "tech_analysis": state.get("tech_analysis", ""),
        "company_analysis": state.get("company_analysis", ""),
        "market_analysis": state.get("market_analysis", ""),
        "missing_items": list(state.get("missing_items") or []),
        "sources": list(state.get("sources") or []),
    }

    print(f"[Save Evaluation] '{startup}' 최종 결과 저장 완료: ...")
    
    return {}   # 기록은 judge가 판정 확정 시 evaluations에 직접 추가


# ─────────────────────────────────────────────────────────────────────────────
# 2. 조건부 라우팅 함수 (Conditional Edges)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_judge(state: GraphState) -> str:
    """
    투자 판단 후 분기:
    - HOLD 판정이고 재시도 전(retry_count == 0)이면 -> 'refine_and_retry'
    - PASS, DROP 또는 1회 재평가 후에도 HOLD이면 -> 'save_evaluation'
    """
    decision = state.get("decision", "")
    retry_count = state.get("retry_count", 0)

    if decision == "HOLD" and retry_count == 0:
        return "refine_and_retry"
    return "save_evaluation"


def route_after_save(state: GraphState) -> str:
    """
    결과 저장 후 분기:
    - 큐에 남은 후보 기업이 있으면 -> 'extract_candidate' (반복 Loop)
    - 5개사 전원 검토 완료되었으면 -> 'reporter' (최종 보고서 작성)
    """
    candidates = state.get("candidates") or []
    if len(candidates) > 0:
        return "extract_candidate"
    return "reporter"


# ─────────────────────────────────────────────────────────────────────────────
# 3. 그래프 구성 및 컴파일
# ─────────────────────────────────────────────────────────────────────────────

def create_investment_graph():
    """
    PDF D.2 Graph 흐름 기준 StateGraph 파이프라인 생성 및 컴파일
    """
    workflow = StateGraph(GraphState)

    # 노드 등록
    workflow.add_node("extract_candidate", extract_candidate)
    workflow.add_node("tech", evaluate_technology)
    workflow.add_node("company", evaluate_company)
    workflow.add_node("market", evaluate_market)
    workflow.add_node("judge", evaluate_judge)
    workflow.add_node("refine_and_retry", refine_and_retry)
    workflow.add_node("save_evaluation", save_evaluation)
    workflow.add_node("reporter", generate_report)

    # 엣지 연결
    # 1. 시작 -> 후보 기업 추출
    workflow.add_edge(START, "extract_candidate")

    # 2. 후보 기업 추출 -> 기술성 -> 기업 -> 시장성 -> 투자 판단 순차 실행
    workflow.add_edge("extract_candidate", "tech")
    workflow.add_edge("tech", "company")
    workflow.add_edge("company", "market")
    workflow.add_edge("market", "judge")

    # 3. 투자 판단 후 분기 (HOLD 시 재검색 루프 vs 결과 저장)
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "refine_and_retry": "refine_and_retry",
            "save_evaluation": "save_evaluation",
        }
    )

    # 4. 재검색 후 다시 기술성 평가 노드로 복귀 (루프)
    workflow.add_edge("refine_and_retry", "tech")

    # 5. 결과 저장 후 분기 (남은 기업 존재 시 반복 vs 보고서 생성)
    workflow.add_conditional_edges(
        "save_evaluation",
        route_after_save,
        {
            "extract_candidate": "extract_candidate",
            "reporter": "reporter",
        }
    )

    # 6. 보고서 생성 -> 종료
    workflow.add_edge("reporter", END)

    # 컴파일
    app = workflow.compile()
    return app


# 전역 실행 객체
graph = create_investment_graph()


# ─────────────────────────────────────────────────────────────────────────────
# 4. 직접 실행 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
    }

    print("🚀 [Graph] Physical AI 스타트업 투자 평가 그래프 실행을 시작합니다...")
    result = graph.invoke(initial_state)
    print("\n🎉 [Graph] 전체 파이프라인 실행이 완료되었습니다!")
    print(f"총 평가된 기업 수: {len(result.get('evaluations', []))}개사")
    print(f"최종 보고서 길이: {len(result.get('report', ''))}자")
