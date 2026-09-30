import operator
from typing import Annotated, Dict, List, TypedDict


# ── PDF C.1 스코어카드 기준 점수 타입 (기본 dict와 100% 호환) ───────────────
class TechScores(TypedDict, total=False):
    tech_differentiation: float  # 기술 차별성 및 특허 (25% 비중)
    purchase_rationale: float  # 구매이유 (10% 비중)


class CompanyScores(TypedDict, total=False):
    founder_competence: float  # 창업자 역량 (20% 비중)
    track_record: float  # 실적 (10% 비중)
    business_model: float  # 수익모델 (10% 비중)


class MarketScores(TypedDict, total=False):
    market_position_and_barriers: float  # 시장 내 포지션 및 진입장벽 (15% 비중)
    market_potential: float  # 시장성 (10% 비중)


# ── PDF D.1 State 설계 (15개 필드 1:1 완벽 일치) ──────────────────────────
class GraphState(TypedDict):
    # 1. candidates | list[str] | 설명: 평가 대기 기업 큐 (5개사)
    #    [기록: 시스템 시작(초기 입력) / 참조: 후보 기업 추출 노드]
    candidates: List[str]

    # 2. current_startup | str | 설명: 현재 평가 대상 기업
    #    [기록: 후보 기업 추출 노드 / 참조: 1~3번 평가 에이전트]
    current_startup: str

    # 3. tech_analysis | str | 설명: 기술, 특허, TRL, 도입 효과 분석
    #    [기록: 기술성 평가 에이전트 / 참조: 투자 판단 에이전트, 보고서 생성 에이전트]
    tech_analysis: str

    # 4. tech_scores | dict | 설명: 기술 차별성/특허, 구매이유 점수 및 근거
    #    [기록: 기술성 평가 에이전트 / 참조: 투자 판단 에이전트]
    tech_scores: TechScores

    # 5. company_analysis | str | 설명: 창업팀, 투자이력, 실적, BM 분석
    #    [기록: 기업 평가 에이전트 / 참조: 투자 판단 에이전트, 보고서 생성 에이전트]
    company_analysis: str

    # 6. company_scores | dict | 설명: 창업자, 실적, 수익모델 점수 및 근거
    #    [기록: 기업 평가 에이전트 / 참조: 투자 판단 에이전트]
    company_scores: CompanyScores

    # 7. market_analysis | str | 설명: 시장 규모, 성장률, 규제, 포지션 분석
    #    [기록: 시장성 평가 에이전트 / 참조: 투자 판단 에이전트, 보고서 생성 에이전트]
    market_analysis: str

    # 8. market_scores | dict | 설명: 시장 포지션, 시장성 점수 및 근거
    #    [기록: 시장성 평가 에이전트 / 참조: 투자 판단 에이전트]
    market_scores: MarketScores

    # 9. missing_items | list[str] | 설명: 근거 미확인(정보 부재) 항목 목록
    #    [기록: 1~3번 평가 에이전트 / 참조: 투자 판단 에이전트, 재검색 노드]
    #    ※ LangGraph 자동 누적(append)을 위해 operator.add 리듀서 적용
    missing_items: Annotated[List[str], operator.add]

    # 10. total_score | float | 설명: 가중 합산 종합 점수 (100점 만점)
    #     [기록: 투자 판단 에이전트 / 참조: 조건부 분기 판정, 보고서 생성 에이전트]
    total_score: float

    # 11. decision | str | 설명: 판정 결과 (PASS, HOLD, DROP)
    #     [기록: 투자 판단 에이전트 / 참조: 조건부 분기 판정, 보고서 생성 에이전트]
    decision: str

    # 12. retry_count | int | 설명: 현재 기업 재심사 횟수 (최대 1회 제한)
    #     [기록: 투자 판단 에이전트 / 참조: 조건부 분기 판정]
    retry_count: int

    # 13. evaluations | list[dict] | 설명: 평가 완료된 기업별 결과 누적
    #     [기록: 투자 판단 에이전트 / 참조: 조건부 분기 판정, 보고서 생성 에이전트]
    #     ※ LangGraph 자동 누적(append)을 위해 operator.add 리듀서 적용
    evaluations: Annotated[List[dict], operator.add]

    # 14. sources | list[str] | 설명: 인용된 참고 문서 출처 누적
    #     [기록: 1~3번 평가 에이전트 / 참조: 보고서 생성 에이전트]
    #     ※ LangGraph 자동 누적(append)을 위해 operator.add 리듀서 적용
    sources: Annotated[List[str], operator.add]

    # 15. report | str | 설명: 최종 투자 심사 보고서 또는 스크리닝 요약서
    #     [기록: 보고서 생성 에이전트 / 참조: 최종 출력]
    report: str
