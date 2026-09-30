"""
[src/agents/tech.py]
우성윤 담당: 기술성 평가 에이전트 (Technology Evaluation Agent)

주요 역할:
1. State에서 current_startup(현재 평가 기업) 확인
2. FAISS 벡터스토어에서 해당 기업의 기술 문서 검색:
   - 필터: company == current_startup AND doc_type IN ["tech", "comprehensive"]
3. prompts/tech_prompt.txt 프롬프트 템플릿에 검색된 문서(Context) 주입
4. GPT-4o-mini(또는 .env에 지정된 OPENAI_MODEL)를 호출하여 심사 수행
5. C.3 Scorecard 기준 채점:
   - 기술 차별성 및 특허 (25%)
   - 구매이유 (10%)
6. GraphState에 결과 반환:
   - tech_analysis: 분석 텍스트
   - tech_scores: {"tech_differentiation": 점수, "purchase_rationale": 점수}
   - missing_items: 정보 부재 항목 리스트 (자동 누적)
   - sources: 인용 출처 리스트 (자동 누적)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from src.state import GraphState, TechScores
from src.vectorstore import build_vectorstore, search_documents

# .env 환경 변수 로드
load_dotenv()

# 프롬프트 템플릿 경로
PROMPT_PATH = Path("prompts/tech_prompt.txt")

# 한글 기업명 <-> 메타데이터 영문 키 매핑
COMPANY_MAP = {
    "트위니": "twinny",
    "twinny": "twinny",
    "뉴빌리티": "neubility",
    "neubility": "neubility",
    "서울로보틱스": "seoulrobotics",
    "seoulrobotics": "seoulrobotics",
    "seoul_robotics": "seoulrobotics",
    "마스오토": "marsauto",
    "marsauto": "marsauto",
    "엑스와이지": "xyz",
    "xyz": "xyz",
}


def normalize_company_name(name: str) -> str:
    """한글/영문 기업명을 벡터스토어 메타데이터 규격으로 정규화합니다."""
    clean = name.strip()
    return COMPANY_MAP.get(clean, clean.lower())


def load_prompt_template() -> str:
    """prompts/tech_prompt.txt 프롬프트 템플릿을 로드합니다."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {PROMPT_PATH}")
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_llm() -> ChatOpenAI:
    """
    OpenAI Chat LLM 인스턴스를 반환합니다.
    기본 모델: gpt-4o-mini (온도: 0.0 객관성 유지)
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=api_key,
    )


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """
    LLM의 응답에서 JSON 블록을 안전하게 추출 및 파싱합니다.
    ```json ... ``` 마크다운 코드블록과 일반 텍스트 모두 대응합니다.
    """
    cleaned = raw_text.strip()

    # 1. ```json ... ``` 코드블록 정규식 추출
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. 첫 번째 '{' 부터 마지막 '}' 까지 추출 시도
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(cleaned[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    # 3. 파싱 실패 시 폴백 기본 구조 반환
    return {
        "tech_analysis": cleaned,
        "tech_scores": {"tech_differentiation": 2.0, "purchase_rationale": 2.0},
        "missing_items": ["LLM JSON 응답 파싱 실패로 인한 기본 점수 처리"],
        "sources": [],
    }


def evaluate_technology(state: GraphState, vectorstore: Optional[Any] = None) -> Dict[str, Any]:
    """
    기술성 평가 에이전트 메인 실행 함수 (LangGraph Node):

    매개변수:
      - state: 현재 파이프라인의 GraphState (current_startup 참조)
      - vectorstore: 사전 로드된 FAISS 벡터스토어 (None일 경우 자동 로드)

    반환값:
      GraphState 업데이트 딕셔너리:
        - tech_analysis: 분석 상세 보고서
        - tech_scores: 스코어카드 점수
        - missing_items: 정보 부재 항목 목록
        - sources: 인용 출처 목록
    """
    startup = state.get("current_startup", "")
    if not startup:
        raise ValueError("GraphState에 'current_startup'이 지정되지 않았습니다.")

    # 1. 기업명 메타데이터 키 정규화
    company_key = normalize_company_name(startup)

    # 2. 벡터스토어 준비 및 검색 실행
    # 필터 조건: company == current_startup AND doc_type IN ["tech", "comprehensive"]
    if vectorstore is None:
        vectorstore = build_vectorstore()

    # 5개사(자율주행, 라이다, 로봇팔, 물류로봇 등) 모두에 공통 적용되는 범용 평가 쿼리
    search_queries = [
        f"{startup} 핵심 기술 솔루션 원천기술 차별성 경쟁우위 TRL 단계",
        f"{startup} 특허 등록 출원 건수 지식재산권 인증",
        f"{startup} 고객 도입 효과 비용 절감 생산성 향상 경제성 수치",
    ]

    docs = []
    seen = set()
    for q in search_queries:
        sub_docs = search_documents(
            vectorstore=vectorstore,
            query=q,
            company=company_key,
            doc_types=["tech", "comprehensive"],
            k=3,
        )
        for d in sub_docs:
            doc_id = (d.metadata.get("file_name"), d.page_content[:60])
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append(d)

    # 3. 검색된 문서 Context 포맷팅
    context_parts = []
    collected_sources = []
    for i, doc in enumerate(docs, 1):
        src_label = doc.metadata.get("source") or doc.metadata.get("file_name") or "문서"
        collected_sources.append(src_label)
        snippet = (
            f"[참고자료 {i}] (출처: {src_label}, 문서유형: {doc.metadata.get('doc_type', '')})\n"
            f"{doc.page_content}\n"
        )
        context_parts.append(snippet)

    context_text = "\n".join(context_parts) if context_parts else "관련 문서를 찾을 수 없습니다."

    # 4. 프롬프트 생성 및 LLM 호출
    prompt_template = load_prompt_template()
    prompt = (
        prompt_template.replace("{startup}", startup)
        .replace("{context}", context_text)
    )

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed_data = parse_llm_json_response(response.content)

    # 5. 스코어 및 결과값 정제
    raw_scores = parsed_data.get("tech_scores", {})
    tech_scores: TechScores = {
        "tech_differentiation": float(
            raw_scores.get("tech_differentiation", raw_scores.get("기술차별성및특허", 2.0))
        ),
        "purchase_rationale": float(
            raw_scores.get("purchase_rationale", raw_scores.get("구매이유", 2.0))
        ),
    }

    tech_analysis = parsed_data.get("tech_analysis", "").strip()
    missing_items = parsed_data.get("missing_items", [])
    if not isinstance(missing_items, list):
        missing_items = [str(missing_items)]

    sources = parsed_data.get("sources", [])
    if not isinstance(sources, list):
        sources = [str(sources)]

    # 문서에서 수집한 출처도 포함 (중복 제거 유지)
    all_sources = list(dict.fromkeys(sources + [s for s in collected_sources if s]))

    print(f"\n[Tech Agent] '{startup}' 기술성 평가 완료:")
    print(f"  - 기술차별성및특허(tech_differentiation): {tech_scores.get('tech_differentiation')}점 / 5.0")
    print(f"  - 구매이유(purchase_rationale): {tech_scores.get('purchase_rationale')}점 / 5.0")
    if missing_items:
        print(f"  - 정보부재 항목: {missing_items}")

    # 6. GraphState 갱신용 딕셔너리 반환
    return {
        "tech_analysis": tech_analysis,
        "tech_scores": tech_scores,
        "missing_items": missing_items,
        "sources": all_sources,
    }


# LangGraph 노드 별칭 등록
tech_node = evaluate_technology
