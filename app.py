"""
[app.py]
AI Startup Investment Evaluation Agent 메인 실행 엔트리포인트
"""

import sys
from dotenv import load_dotenv
from src.graph import graph
from src.state import GraphState

load_dotenv()


def main():
    print("=" * 60)
    print("AI 스타트업 투자 평가 파이프라인 (Physical AI / Robotics)")
    print("=" * 60)

    # 평가 대상 5개사
    candidates = ["트위니", "뉴빌리티", "서울로보틱스", "마스오토", "엑스와이지"]

    initial_state: GraphState = {
        "candidates": candidates,
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

    try:
        final_state = graph.invoke(initial_state)
        print("\n" + "=" * 60)
        print("🎉 전체 스타트업 투자 평가가 성공적으로 완료되었습니다!")
        print("=" * 60)
        evaluations = final_state.get("evaluations", [])
        for ev in evaluations:
            startup = ev.get("startup")
            score = ev.get("total_score")
            decision = ev.get("decision")
            print(f"- {startup}: {score}점 [{decision}]")

    except Exception as e:
        print(f"\n❌ 실행 중 오류가 발생했습니다: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
