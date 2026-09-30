# -*- coding: utf-8 -*-
"""
시장성 평가 에이전트 테스트
==========================
벡터DB와 OpenAI API 없이 파싱·검증 로직만 확인한다.
가짜 검색기와 가짜 LLM을 주입해 검증 규칙이 실제로 동작하는지 본다.

실행
    python -m tests.test_market_agent
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import market


# ── 가짜 부품 ────────────────────────────────────────────────────────────────
class FakeDoc:
    def __init__(self, title: str, content: str, url: str = ""):
        self.metadata = {"title": title, "doc_type": "market",
                         "company": "common", "url": url}
        self.page_content = content


MARKET_DOCS = [
    FakeDoc(
        "Mordor_물류자동화시장.pdf",
        "글로벌 물류 자동화 시장은 2025년 816억 5천만 달러에서 "
        "2031년 1,327억 4천만 달러로 성장할 전망이다. CAGR은 7.91%다.",
        "https://www.mordorintelligence.kr/industry-reports/logistics-automation-market",
    ),
    FakeDoc(
        "첨단로봇_규제혁신방안.pdf",
        "지능형로봇법 개정으로 실외 이동 로봇의 정의와 안전성 기준이 신설됐다. "
        "생활물류서비스산업발전법 개정으로 배송 대행 운송 수단으로 로봇을 활용할 수 있다.",
    ),
]


class FakeStore:
    """search_documents가 대신 호출되므로 실제로는 쓰이지 않지만 형태를 맞춰 둔다."""
    pass

def make_fake_llm(response: str):
    class FakeLLM:
        def invoke(self, prompt):
            class R:
                content = response
            return R()
    return FakeLLM()


def patch(monkey_docs, response):
    """market 모듈의 외부 의존성을 가짜로 바꾼다."""
    market._get_vectorstore = lambda: FakeStore()
    market.retrieve_market_docs = lambda vs, startup, k=6: monkey_docs
    market.ChatOpenAI = lambda **kwargs: make_fake_llm(response)


# ── 테스트 ───────────────────────────────────────────────────────────────────
def test_정상_응답():
    """근거가 문서에 있으면 모델 점수를 그대로 인정한다."""
    patch(MARKET_DOCS, (
        '[분석]\n물류 자동화 시장은 "2031년 1,327억 4천만 달러로 성장할 전망" 이며 '
        'CAGR은 "7.91%"다. 규제는 우호적이다. CAGR이 20% 미만이라 시장성은 3점으로 둔다.\n\n'
        "[점수]\n시장내포지션및진입장벽: 4\n시장성: 3\n\n"
        "[정보부재항목]\n없음\n\n"
        "[출처]\nMordor_물류자동화시장.pdf\n"
    ))
    out = market.evaluate_market({"current_startup": "서울로보틱스"})

    assert out["market_scores"]["market_position_and_barriers"] == 4.0
    assert out["market_scores"]["market_potential"] == 3.0
    assert out["missing_items"] == []
    assert any("Mordor_물류자동화시장.pdf" in s for s in out["sources"])
    assert any("http" in s for s in out["sources"])
    print("✅ 정상 응답: 점수 그대로 유지")


def test_지어낸_인용은_점수_제한():
    """문서에 없는 인용을 넣으면 최대 4점으로 깎는다."""
    patch(MARKET_DOCS, (
        '[분석]\n시장은 "연평균 45% 성장하며 2030년 500조 원" 규모다.\n\n'
        "[점수]\n시장내포지션및진입장벽: 5\n시장성: 5\n\n"
        "[정보부재항목]\n없음\n\n"
        "[출처]\n존재하지않는리포트.pdf\n"
    ))
    out = market.evaluate_market({"current_startup": "서울로보틱스"})

    assert out["market_scores"]["market_position_and_barriers"] == 4.0
    assert out["market_scores"]["market_potential"] == 4.0
    assert "존재하지않는리포트.pdf" not in out["sources"]   # 가짜 출처는 버린다
    assert "코드 보정" in out["market_analysis"]
    print("✅ 지어낸 인용: 5점 → 4점 하향, 가짜 출처 제거")


def test_정보부재면_2점():
    """TAM·CAGR을 못 찾았다고 하면 시장성을 2점으로 강제한다."""
    patch(MARKET_DOCS, (
        '[분석]\n시장 규모 자료를 찾지 못했다.\n\n'
        "[점수]\n시장내포지션및진입장벽: 3\n시장성: 4\n\n"
        "[정보부재항목]\nTAM 규모, CAGR 수치\n\n"
        "[출처]\nMordor_물류자동화시장.pdf\n"
    ))
    out = market.evaluate_market({"current_startup": "마스오토"})

    assert out["market_scores"]["market_potential"] == 2.0
    assert len(out["missing_items"]) >= 2
    print("✅ 정보 부재: 시장성 4점 → 2점")


def test_검색결과_0건():
    """문서를 못 찾으면 두 항목 모두 2점으로 처리한다."""
    patch([], "[분석]\n아무 내용\n\n[점수]\n시장내포지션및진입장벽: 5\n시장성: 5\n")
    out = market.evaluate_market({"current_startup": "엑스와이지"})

    assert out["market_scores"]["market_position_and_barriers"] == 2.0
    assert out["market_scores"]["market_potential"] == 2.0
    assert out["sources"] == []
    print("✅ 검색 0건: 두 항목 모두 2점")


def test_형식이_깨져도_안_멈춘다():
    """출력 형식이 엉망이어도 예외 없이 2점으로 넘어간다."""
    patch(MARKET_DOCS, "시장이 좋아 보입니다. 점수는 알아서 판단하세요.")
    out = market.evaluate_market({"current_startup": "뉴빌리티"})

    assert out["market_scores"]["market_position_and_barriers"] == 2.0
    assert out["market_scores"]["market_potential"] == 2.0
    assert any("파싱 실패" in m for m in out["missing_items"])
    print("✅ 형식 파손: 예외 없이 2점 처리")


if __name__ == "__main__":
    test_정상_응답()
    test_지어낸_인용은_점수_제한()
    test_정보부재면_2점()
    test_검색결과_0건()
    test_형식이_깨져도_안_멈춘다()
    print("\n테스트 5개 모두 통과")