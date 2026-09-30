from typing import Any, Dict
from src.state import GraphState


def evaluate_market(state: GraphState) -> Dict[str, Any]:
    """김지훈 담당: 시장성 평가 에이전트 (임시 스텁)"""
    return {
        "market_analysis": "시장성 분석 더미 내용",
        "market_scores": {
            "market_position_and_barriers": 4.0,
            "market_potential": 4.0,
        },
    }
