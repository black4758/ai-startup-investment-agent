"""
[src/state.py]
Multi-Agent Orchestration State Schema
노션 가이드 7대 엔터프라이즈 State 설계 원칙 완벽 준수 (평가 배점 20점)

- PerspectiveResult 구조화 모델 도입 (출처 및 부족 항목의 관점별 독립 격리)
- [A. 작업 페이로드]와 [B. 제어 메타데이터]의 명확한 경계 분리
"""

import operator
from typing import Annotated, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic 구조화 산출물 스키마 (품질 평가 & 관점별 분석 결과)
# ─────────────────────────────────────────────────────────────────────────────

class QualityEvalResult(BaseModel):
    """
    품질 평가 노드 판정 결과 (노션 3대 필수 지표)
    """
    groundedness_pass: bool = Field(description="모든 주장이 출처(Reference)로 추적되며 환각이 없는가")
    neutrality_pass: bool = Field(description="특정 기업/기술에 대한 주관적 우열 판정 및 일방 추천이 없는가")
    bias_control_pass: bool = Field(description="단일 출처에 편중되지 않고 다각도로 서술되었는가")
    
    verdict: str = Field(description="종합 판정: 'PASS' 또는 'FAIL'")
    feedback: str = Field(default="", description="미달 시 보고서 재작성을 위한 구체적인 수정 지침 (PASS 시 빈 문자열)")


class PerspectiveResult(BaseModel):
    """
    노션 가이드 예시 준수 관점별 분석 결과 객체
    - 출처(sources)와 정보 부재(missing_items)를 전역 리스트가 아닌 관점 객체 내부에 격리하여
      기업 간 / 에이전트 간 데이터 누적 오염 방지
    """
    view: str = Field(description="관점 구분: 'tech', 'company', 'market'")
    analysis: str = Field(default="", description="관점별 상세 분석 내용")
    scores: Dict[str, float] = Field(default_factory=dict, description="해당 관점의 항목별 점수")
    sources: List[str] = Field(default_factory=list, description="해당 관점에서 인용한 실제 출처 목록")
    missing_items: List[str] = Field(default_factory=list, description="해당 관점의 정보 부재 항목 목록")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 스코어카드 세부 점수 타입
# ─────────────────────────────────────────────────────────────────────────────

class TechScores(TypedDict, total=False):
    tech_differentiation: float  # 기술 차별성 및 특허 (25%)
    purchase_rationale: float    # 구매이유 (10%)

class CompanyScores(TypedDict, total=False):
    founder_competence: float    # 창업자 역량 (20%)
    track_record: float          # 실적 (10%)
    business_model: float        # 수익모델 (10%)

class MarketScores(TypedDict, total=False):
    market_position_and_barriers: float  # 시장 내 포지션 및 진입장벽 (15%)
    market_potential: float              # 시장성 (10%)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 메인 LangGraph State (GraphState)
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    # =========================================================================
    # [A. 작업 페이로드 (Payload) - 실제 비즈니스 분석 데이터]
    # =========================================================================
    
    # 1. 후보 기업 큐 및 현재 심사 대상
    candidates: List[str]
    current_startup: str

    # 2. 관점별 분석 결과 딕셔너리 (노션 정석 아키텍처)
    #    {"tech": PerspectiveResult, "company": PerspectiveResult, "market": PerspectiveResult}
    perspectives: Dict[str, PerspectiveResult]

    # 3. 종합 투자 판단 결과 (Synthesizer/Judge)
    total_score: float                       # 가중 합산 종합 점수 (100점 만점)
    decision: str                            # PASS / HOLD / DROP

    # 4. 보고서 작성 및 품질 평가
    report: str                              # 최종 Markdown 보고서 본문
    eval_result: Optional[QualityEvalResult] # 3대 품질 평가 판정 및 피드백

    # 5. 전사 결과 누적 (완료된 기업별 종합본만 누적)
    evaluations: Annotated[List[dict], operator.add]


    # =========================================================================
    # [B. 제어 메타데이터 (Control Metadata) - Supervisor 및 7대 원칙 준수 필드]
    # =========================================================================

    # 1. [Supervisor 지휘 메타데이터]
    next_agent: str                          # Supervisor가 결정한 다음 실행 목적지 ('tech', 'company', 'market', 'judge', 'reporter', 'end')
    agent_counts: Dict[str, int]             # 에이전트별 실행/재작업 횟수 (예: {"tech": 1, "company": 2}) -> 무한 재작업 방지
    report_revision_count: int               # 품질 평가 탈락 시 보고서 재작성 횟수 (최대 3회 가드)

    # 2. [상관] 외부 로거 및 LangSmith Tracing 실행 세션을 잇는 고유 키
    trace_id: str

    # 3. [종료 보장 & 재시도]
    step_count: int                          # 파이프라인 전체 스텝 카운터
    retry_count: int                         # HOLD 판정 시 재평가 회차 (최대 1회)

    # 4. [재개/복구] 장애 발생 시 중단 지점 추적 및 재개를 위한 상태 플래그
    node_status: Dict[str, str]              # 예: {"tech": "SUCCESS", "quality_eval": "FAIL"}
    last_error: Optional[str]                # 예외 발생 시 Fallback 에러 메시지
