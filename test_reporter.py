import os
from dotenv import load_dotenv
from src.agents.reporter import report_agent

# .env 파일에서 OPENAI_API_KEY 로드
load_dotenv()

# 테스트용 가짜(Mock) State 데이터
mock_state = {
    "evaluations": [
        {"startup": "엑스와이지", "decision": "PASS", "total_score": 85.0},
        {"startup": "트위니", "decision": "HOLD", "total_score": 70.0}
    ],
    "tech_analysis": "엑스와이지는 로봇 팔 기반 F&B 자동화 기술을 상용화하였으며, 특허 5건을 보유하고 TRL 9단계에 도달함.",
    "company_analysis": "창업팀은 로봇 공학 및 디자인 전문가로 구성되었으며, 주요 대기업 사내 카페에 반복 납품 실적 보유.",
    "market_analysis": "서비스 로봇 시장은 연평균 성장률 20% 이상을 기록 중이며, 엑스와이지는 매장 자동화 분야에서 선도적 포지션을 확보함.",
    "sources": ["xyz_tech_whitepaper.pdf", "xyz_news_2026.txt", "xyz_news_2026.txt"]
}

print("보고서 생성 노드 실행 중...")
result_state = report_agent(mock_state)

print("✅ 테스트 완료!")
print("출력된 보고서 내용 미리보기:")
print("-" * 50)
print(result_state["report"][:500] + "\n...[중략]...")
print("-" * 50)
print("outputs 폴더에 마크다운 파일이 생성되었는지 확인하세요.")
