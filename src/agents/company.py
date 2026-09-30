from typing import Any, Dict
from src.state import GraphState


def evaluate_company(state: GraphState) -> Dict[str, Any]:
    """이진서 담당: 기업 역량 평가 에이전트 (임시 스텁)"""
    return {
        "company_analysis": "기업 분석 더미 내용",
        "company_scores": {
            "founder_competence": 4.0,
            "track_record": 3.0,
            "business_model": 3.0,
        },
    }
