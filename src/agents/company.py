"""
[src/agents/company.py]
이진서 담당: 기업 평가 에이전트 (Company Evaluation Agent)

주요 역할:
1. State에서 current_startup(현재 평가 기업) 확인
2. FAISS 벡터스토어에서 해당 기업의 기업 문서 검색 (PDF B.2):
   - 필터: company == current_startup AND doc_type IN ["company", "news", "comprehensive"]
   - 평가 항목(창업자/실적/수익모델)별 쿼리 + top-5
   - 1차 후보 20개 중 참고문헌 청크 제외, 소제목이 평가 항목과 맞는 청크 우선 재정렬
3. prompts/company_prompt.txt 프롬프트 템플릿에 검색된 문서(Context) 주입
4. GPT-4o-mini(또는 .env에 지정된 OPENAI_MODEL)를 호출하여 심사 수행
5. C.3 Scorecard 기준 채점:
   - 창업자 (20%)
   - 실적 (10%)
   - 수익모델 (10%)
6. C.4 정보 부재 규칙을 코드로 재검증:
   - 근거 유형 "없음" -> 2점 + 정보 부재 기록
   - 근거 유형 "회사발표" -> 최대 4점
7. HOLD 재평가(retry_count >= 1) 시 1차 평가에서 2점 이하였던 항목만 보강 쿼리로 재검색 (PDF C.5)
8. GraphState에 결과 반환:
   - company_analysis: 분석 텍스트
   - company_scores: {"founder_competence": 점수, "track_record": 점수, "business_model": 점수}
   - missing_items: 정보 부재 항목 리스트 (자동 누적)
   - sources: 인용 출처 리스트 (자동 누적)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage

from src.state import CompanyScores, GraphState
from src.vectorstore import build_vectorstore, search_documents
from src.agents.tech import get_llm, normalize_company_name

# 프롬프트 템플릿 경로
PROMPT_PATH = Path("prompts/company_prompt.txt")

# 검색 필터 문서 유형 및 top-k (PDF B.2 에이전트별 검색 필터)
DOC_TYPES = ["company", "news", "comprehensive"]
TOP_K = 5
FETCH_K = 20  # 소제목 재정렬 전 1차 후보 수

# 소제목(헤더 메타데이터) 재정렬용 항목별 키워드
HEADER_KEYWORDS = {
    "창업자": ["창업", "대표", "경영진", "팀", "인력", "CEO", "CTO"],
    "실적": ["매출", "재무", "실적", "고객", "계약", "손익", "지표"],
    "수익모델": ["수익", "사업 모델", "BM", "RaaS", "비즈니스", "과금"],
}

# 근거가 아닌 청크(참고문헌 목록, 검색 평가용 정답 세트) 제외 키워드
EXCLUDE_KEYWORDS = ["참고문헌", "REFERENCE", "정답 세트"]

# 정보 부재 점수 / 회사 발표 상한 (PDF C.4)
MISSING_SCORE = 2.0
COMPANY_CLAIM_MAX = 4.0

# 점수 키 <-> 한글 항목명 매핑 (missing_items 기록 및 한글 키 응답 대비)
SCORE_KEYS = {
    "founder_competence": "창업자",
    "track_record": "실적",
    "business_model": "수익모델",
}

# 1차 평가용 항목별 검색 쿼리
BASE_QUERIES = {
    "창업자": "{startup} 창업자 대표이사 학력 경력 공동창업 경영진 팀 구성",
    "실적": "{startup} 매출액 성장률 영업손익 주요 고객사 계약 납품 운영 실적",
    "수익모델": "{startup} 수익 모델 사업 모델 구독 RaaS 렌탈 유지보수 반복 매출",
}

# HOLD 재평가용 보강 쿼리 (정보 부재 항목별)
RETRY_QUERIES = {
    "창업자": [
        "{startup} 창업 배경 창업자 이력 박사 연구 경험 엑싯",
        "{startup} CTO 기술 총괄 핵심 인력 임직원 수 경영진 교체",
    ],
    "실적": [
        "{startup} 연도별 매출 추이 전년 대비 재무제표 손익 공시",
        "{startup} 대기업 고객 공급 계약 체결 상용화 도입 사례 운영 대수",
    ],
    "수익모델": [
        "{startup} 과금 방식 월 구독료 서비스 이용료 가격 정책",
        "{startup} 관제 소프트웨어 플랫폼 유지보수 계약 수익원",
    ],
}

# 5개사 평가 동안 벡터스토어(bge-m3 모델 포함)를 한 번만 로드
_vectorstore_cache: Optional[Any] = None


def get_vectorstore() -> Any:
    """벡터스토어를 한 번만 로드하여 재사용합니다."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        _vectorstore_cache = build_vectorstore()
    return _vectorstore_cache


def load_prompt_template() -> str:
    """prompts/company_prompt.txt 프롬프트 템플릿을 로드합니다."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {PROMPT_PATH}")
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_weak_items(prev_scores: Dict[str, Any]) -> List[str]:
    """
    현재 기업의 1차 평가 점수(company_scores)에서 정보 부재 수준(2점 이하) 항목을 반환합니다.
    - missing_items는 누적 필드라 다른 기업·다른 에이전트 항목이 섞이므로,
      기업이 바뀔 때 초기화되는 company_scores를 기준으로 재검색 대상을 정합니다.
    """
    weak = []
    for key, kor in SCORE_KEYS.items():
        try:
            if float(prev_scores.get(key, 5.0)) <= MISSING_SCORE:
                weak.append(kor)
        except (TypeError, ValueError):
            weak.append(kor)
    return weak


def build_queries(startup: str, weak_items: List[str]) -> List[Tuple[str, str]]:
    """
    (평가 항목, 검색 쿼리) 목록을 생성합니다.
    - 1차 평가: 항목별 기본 쿼리 3개
    - HOLD 재평가: 기본 쿼리 + 1차 평가에서 정보 부재였던 항목의 보강 쿼리
    """
    queries = [(item, q.format(startup=startup)) for item, q in BASE_QUERIES.items()]

    for item in weak_items:
        queries.extend((item, q.format(startup=startup)) for q in RETRY_QUERIES.get(item, []))

    return queries


def _section_title(doc: Any) -> str:
    """청크의 가장 하위 소제목을 반환합니다. (MarkdownHeaderTextSplitter 메타데이터)"""
    meta = doc.metadata
    return meta.get("Header 3") or meta.get("Header 2") or meta.get("Header 1") or ""


def retrieve_for_item(vectorstore: Any, company_key: str, item: str, query: str) -> List[Any]:
    """
    항목별 문서 검색:
    1. 메타데이터 필터(company, doc_type)로 1차 후보 FETCH_K개 검색
    2. 참고문헌 목록 등 근거가 아닌 청크 제외
    3. 소제목에 평가 항목 키워드가 있는 청크를 앞으로 재정렬 후 top-5 반환
    """
    candidates = search_documents(
        vectorstore=vectorstore,
        query=query,
        company=company_key,
        doc_types=DOC_TYPES,
        k=FETCH_K,
    )

    candidates = [
        d for d in candidates
        if not any(ex in _section_title(d) + d.page_content[:80] for ex in EXCLUDE_KEYWORDS)
    ]

    keywords = HEADER_KEYWORDS.get(item, [])
    matched = [d for d in candidates if any(kw in _section_title(d) for kw in keywords)]
    others = [d for d in candidates if d not in matched]

    return (matched + others)[:TOP_K]


def _parse_refs(values: Any, num_docs: int) -> List[int]:
    """LLM이 적은 참고자료 번호 목록에서 유효한 번호(1~num_docs)만 정수로 추출합니다."""
    if not isinstance(values, list):
        values = [values]
    refs = []
    for v in values:
        match = re.search(r"\d+", str(v))
        if match and 1 <= int(match.group()) <= num_docs:
            refs.append(int(match.group()))
    return refs


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """
    LLM의 응답에서 JSON 블록을 안전하게 추출 및 파싱합니다.
    ```json ... ``` 마크다운 코드블록과 일반 텍스트 모두 대응합니다.
    """
    cleaned = raw_text.strip()

    # 1. ```json ... ``` 코드블록 정규식 추출
    json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
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

    # 3. 파싱 실패 시 폴백 기본 구조 반환 (전 항목 정보 부재 처리)
    return {
        "company_analysis": cleaned,
        "evaluation": {
            key: {"basis": "LLM JSON 응답 파싱 실패", "type": "없음", "score": MISSING_SCORE}
            for key in SCORE_KEYS
        },
        "missing_items": list(SCORE_KEYS.values()),
    }


def apply_scoring_rules(
    evaluation: Dict[str, Any],
    missing_items: List[str],
) -> Tuple[CompanyScores, List[str]]:
    """
    LLM 항목별 평가(basis -> type -> criterion -> score)에 PDF C.4 정보 부재 처리 규칙을 코드로 재적용합니다.
    - 근거 유형 "없음" 또는 LLM이 정보 부재로 기록한 항목 -> 2점
    - 근거 유형 "회사발표" -> 최대 4점
    - 그 외 -> 1~5점 범위로 보정
    """
    scores: CompanyScores = {}
    missing = list(missing_items)

    for key, kor in SCORE_KEYS.items():
        ev = evaluation.get(key) or {}
        if not isinstance(ev, dict):
            ev = {}

        try:
            value = float(ev.get("score", MISSING_SCORE))
        except (TypeError, ValueError):
            value = MISSING_SCORE
        value = min(max(value, 1.0), 5.0)

        ev_type = str(ev.get("type", "")).replace(" ", "")
        marked_missing = any(kor in m for m in missing)

        if ev_type == "없음" or marked_missing:
            value = MISSING_SCORE
            if not marked_missing:
                missing.append(kor)
        elif ev_type == "회사발표":
            value = min(value, COMPANY_CLAIM_MAX)

        scores[key] = value

    return scores, list(dict.fromkeys(missing))


def evaluate_company(state: GraphState, vectorstore: Optional[Any] = None) -> Dict[str, Any]:
    """
    기업 평가 에이전트 메인 실행 함수 (LangGraph Node):

    매개변수:
      - state: 현재 파이프라인의 GraphState (current_startup, retry_count, company_scores 참조)
      - vectorstore: 사전 로드된 FAISS 벡터스토어 (None일 경우 자동 로드)

    반환값:
      GraphState 업데이트 딕셔너리:
        - company_analysis: 분석 상세 보고서
        - company_scores: 스코어카드 점수
        - missing_items: 정보 부재 항목 목록
        - sources: 인용 출처 목록
    """
    startup = state.get("current_startup", "")
    if not startup:
        raise ValueError("GraphState에 'current_startup'이 지정되지 않았습니다.")

    retry_count = state.get("retry_count", 0)
    # HOLD 재평가 시 1차 평가에서 정보 부재였던 항목 (PDF C.5)
    weak_items = get_weak_items(state.get("company_scores") or {}) if retry_count >= 1 else []

    # 1. 기업명 메타데이터 키 정규화
    company_key = normalize_company_name(startup)

    # 2. 벡터스토어 준비 및 검색 실행
    # 필터 조건: company == current_startup AND doc_type IN ["company", "news", "comprehensive"]
    if vectorstore is None:
        vectorstore = get_vectorstore()

    docs = []
    seen = set()
    for item, q in build_queries(startup, weak_items):
        for d in retrieve_for_item(vectorstore, company_key, item, q):
            doc_id = (d.metadata.get("file_name"), d.page_content[:60])
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append(d)

    # 3. 검색된 문서 Context 포맷팅
    context_parts = []
    doc_sources = []  # 참고자료 번호(1부터) 순서대로 "출처 표기 (발행일). URL" 기록 (REFERENCE 작성용)
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src_label = meta.get("reference") or meta.get("source") or meta.get("file_name") or "문서"
        date = meta.get("date", "")
        url = meta.get("url", "")

        reference = f"{src_label} ({date})" if date and date not in src_label else src_label
        if url:
            reference += f". {url}"
        doc_sources.append(reference)

        context_parts.append(
            f"[참고자료 {i}] (출처: {src_label}, 발행일: {date}, "
            f"문서유형: {doc.metadata.get('doc_type', '')})\n"
            f"{doc.page_content}\n"
        )

    context_text = "\n".join(context_parts) if context_parts else "관련 문서를 찾을 수 없습니다."

    # 4. 프롬프트 생성 및 LLM 호출
    prompt = (
        load_prompt_template()
        .replace("{startup}", startup)
        .replace("{context}", context_text)
    )

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed_data = parse_llm_json_response(response.content)

    # 5. 스코어 정제 및 C.4 규칙 재검증
    evaluation = parsed_data.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        evaluation = {}

    # LLM이 기록한 정보 부재 항목을 기업 평가 3개 항목명으로 정규화
    llm_missing = parsed_data.get("missing_items", [])
    if not isinstance(llm_missing, list):
        llm_missing = [str(llm_missing)]
    llm_missing = [kor for kor in SCORE_KEYS.values() if any(kor in str(m) for m in llm_missing)]

    company_scores, missing_items = apply_scoring_rules(evaluation, llm_missing)

    # 항목별 채점 근거와 근거 섹션을 분석 텍스트에 덧붙여 판단/보고서 에이전트가 감점 사유를 참조할 수 있게 함
    company_analysis = str(parsed_data.get("company_analysis", "")).strip()
    rationale_lines = []
    used_refs: List[int] = []  # 항목별로 LLM이 근거로 명시한 참고자료 번호
    for key, kor in SCORE_KEYS.items():
        ev = evaluation.get(key) or {}
        if not isinstance(ev, dict):
            continue
        refs = _parse_refs(ev.get("refs", []), len(docs))
        used_refs.extend(refs)
        if ev.get("basis"):
            # 참고자료 번호는 에이전트 내부에서만 의미가 있으므로 소제목으로 바꿔 기록
            sections = ", ".join(dict.fromkeys(_section_title(docs[r - 1]) or "본문" for r in refs)) or "-"
            rationale_lines.append(
                f"- {kor} {company_scores[key]}점 [{ev.get('type', '-')}]: {ev.get('basis')} "
                f"/ {ev.get('criterion', '')} (근거 섹션: {sections})"
            )
    if rationale_lines:
        company_analysis += "\n\n[항목별 채점 근거]\n" + "\n".join(rationale_lines)

    # 항목별 근거로 쓰인 참고자료만 출처로 기록 (REFERENCE는 실제 활용 자료만), 번호 인용이 없으면 검색된 전체 문서로 대체
    used_sources = [doc_sources[r - 1] for r in used_refs]
    all_sources = list(dict.fromkeys(used_sources or doc_sources))

    # 재평가 시 1차 평가에서 이미 기록된 정보 부재 항목은 중복 누적하지 않음 (missing_items는 누적 필드)
    reported_missing = [m for m in missing_items if m not in weak_items]

    tag = f" (HOLD 재평가 {retry_count}회차)" if retry_count >= 1 else ""
    print(f"\n[Company Agent] '{startup}' 기업 평가 완료{tag}:")
    for key, kor in SCORE_KEYS.items():
        ev = evaluation.get(key) or {}
        ev_type = ev.get("type", "-") if isinstance(ev, dict) else "-"
        print(f"  - {kor}({key}): {company_scores.get(key)}점 / 5.0 [근거: {ev_type}]")

    # 6. GraphState 갱신용 딕셔너리 반환
    return {
        "company_analysis": company_analysis,
        "company_scores": company_scores,
        "missing_items": reported_missing,
        "sources": all_sources,
    }


# LangGraph 노드 별칭 등록
company_node = evaluate_company
