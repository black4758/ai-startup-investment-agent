
from dotenv import load_dotenv
from unittest.mock import patch
from src.agents.reporter import generate_report
from src.state import QualityEvalResult

load_dotenv()

# 실제 기업 데이터가 아닌 테스트용 가상 데이터
test_state = {
    "evaluations": [
        {
            "company": "Test Robotics",
            "total_score": 85,
            "decision": "PASS"
        }
    ],
    "tech_analysis": "로봇 제어 기술을 개발 중이나 기술 성숙도 검증 자료는 부족함.",
    "company_analysis": "창업자 경력과 매출 실적에 대한 검증 자료가 부족함.",
    "market_analysis": "산업용 로봇 시장을 목표로 하나 시장 규모에 대한 검증 자료는 없음.",
    "sources": ["테스트용 내부 분석 자료"],
    "report": "",
    "eval_result": None
}

print("=== 최초 보고서 생성 테스트 ===")

result = generate_report(test_state)

assert isinstance(result["report"], str)

assert result["report"].strip()

print("보고서 생성 성공")
print("보고서 길이:", len(result["report"]))

# FAIL 재작성 테스트
test_state["report"] = result["report"]

test_state["eval_result"] = QualityEvalResult(
    groundedness_pass=False,
    neutrality_pass=True,
    bias_control_pass=True,
    verdict="FAIL",
    feedback="보고서의 '5. 리스크와 한계' 항목에 '본 평가는 제공된 제한적 자료에 기반하므로 추가 실사가 필요함.'이라는 문장을 반드시 추가하세요."
)

print("\n=== FAIL 재작성 테스트 ===")


revised_result = generate_report(test_state)

assert isinstance(revised_result["report"], str)
assert revised_result["report"].strip()

print("재작성 실행 성공")
print("재작성 보고서 길이:", len(revised_result["report"]))
print("기존 보고서와 내용이 다른가:",
      result["report"] != revised_result["report"])
required_sentence = "본 평가는 제공된 제한적 자료에 기반하므로 추가 실사가 필요함."

print(
    "피드백 문장 반영 여부:",
    required_sentence in revised_result["report"]
)