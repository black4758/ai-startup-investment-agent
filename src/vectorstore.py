"""
[src/vectorstore.py]
피지컬 AI 스타트업 투자 평가 에이전트 시스템을 위한 통합 벡터스토어 모듈

주요 기능:
1. 문서 자동 로드:
   - data/ 폴더 하위의 모든 .md 및 .pdf 파일 재귀적 로드 (참조용 _references* 파일 제외)
   - [추가된 마켓 문서] data/market/spri_physical_ai_report.md (SPRi 피지컬 AI 이슈리포트) 포함
2. 파일 경로 및 내용 기반 메타데이터 자동 태깅:
   - company: twinny / neubility / seoulrobotics / marsauto / xyz / common (시장 문서는 common)
   - doc_type: tech / company / news / market / comprehensive
   - source: 발행기관, 언론사 또는 공식 출처
   - date: 발행일 (YYYY-MM-DD 또는 YYYY-MM)
3. 소제목 기반 지능형 청킹:
   - Markdown: MarkdownHeaderTextSplitter로 소제목(#, ##, ###) 단위 1차 분할 후, 1,500자 초과 시 재분할
   - PDF: PyPDFLoader 기반 로딩 후 1,200자 단위 분할
4. BAAI/bge-m3 고성능 다국어 임베딩 (1024차원)
5. FAISS 벡터 DB 구축 및 로컬 캐싱 (storage/faiss_index):
   - 최초 1회 인덱싱 후 재실행 시 디스크 캐시에서 수 초 내 로드
6. 에이전트 검색 필터 완벽 지원:
   - dict 필터 내 리스트 값 지원 (예: doc_type IN ['tech', 'comprehensive'])
   - callable 필터 지원
"""

import os
import re
import glob
from typing import List, Dict, Any, Optional, Union
import torch

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


# 기본 임베딩 모델 및 FAISS 인덱스 저장 디렉토리 상수
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_INDEX_DIR = "storage/faiss_index"


def get_device() -> str:
    """
    현재 실행 환경에 최적화된 컴퓨팅 디바이스를 반환합니다.
    - Apple Silicon(M1/M2/M3/M4 Mac): 'mps' 가속 사용
    - 기타 환경: 'cpu' 사용
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    BAAI/bge-m3 기반의 HuggingFaceEmbeddings 인스턴스를 생성합니다.
    - 한국어/영어 혼용 및 딥테크 전문 용어에 강한 1024차원 다국어 임베딩 모델
    - 코사인 유사도 계산을 위한 L2 정규화(normalize_embeddings=True) 활성화
    """
    device = get_device()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_metadata(file_path: str, content: str = "") -> Dict[str, Any]:
    """
    파일 경로(Path) 및 문서 내용(Frontmatter/본문)을 분석하여 메타데이터를 자동 추출합니다.

    추출 항목:
      - company: twinny / neubility / seoulrobotics / marsauto / xyz / common
      - doc_type: tech / company / news / market / comprehensive
      - source: 발행기관, 언론사, 또는 출처 표기 (보고서 REFERENCE 작성용)
      - date: 발행일 (YYYY-MM-DD 또는 YYYY-MM)
      - file_name: 파일명
      - file_path: 파일 전체 상대 경로
    """
    rel_path = os.path.relpath(file_path, "data").lower()
    filename = os.path.basename(file_path).lower()

    # ─────────────────────────────────────────────────────────────────
    # 1. 기업명(company) 태깅 로직
    # ─────────────────────────────────────────────────────────────────
    if "twinny" in rel_path:
        company = "twinny"
    elif "neubility" in rel_path or "newbility" in rel_path:
        company = "neubility"
    elif "seoul" in rel_path or "robotics" in rel_path:
        company = "seoulrobotics"
    elif "xyz" in rel_path:
        company = "xyz"
    elif "marsauto" in rel_path:
        # 마스오토 폴더 내의 공통 시장 보고서는 common으로 분류
        if "market_autonomous_driving" in filename:
            company = "common"
        else:
            company = "marsauto"
    elif "market" in rel_path:
        # data/market/ 하위 문서(SPRi 리포트, KIRIA 리포트)는 모두 공통(common)으로 태깅
        company = "common"
    else:
        company = "common"

    # ─────────────────────────────────────────────────────────────────
    # 2. 문서 유형(doc_type) 태깅 로직
    # ─────────────────────────────────────────────────────────────────
    if "market" in filename or "market" in rel_path.split(os.sep):
        # spri_physical_ai_report.md, kiria_logistics_robot_market_report.md 등
        doc_type = "market"
    elif "tech" in filename or "whitepaper" in filename:
        doc_type = "tech"
    elif "news" in filename:
        doc_type = "news"
    elif "financial" in filename:
        doc_type = "company"
    elif "profile" in filename or "business" in filename:
        # 뉴빌리티 프로필은 기술/재무/특허가 모두 포함된 종합 문서이므로 comprehensive 지정
        if company == "neubility":
            doc_type = "comprehensive"
        else:
            doc_type = "company"
    elif "research_note" in filename or "corporate_analysis" in filename or "report" in filename:
        # 서울로보틱스 조사노트, 엑스와이지 분석 보고서 등은 기술/기업 전반을 다룸
        doc_type = "comprehensive"
    else:
        doc_type = "company"

    source = ""
    date = ""

    # ─────────────────────────────────────────────────────────────────
    # 3. YAML Frontmatter가 존재하는 마크다운 문서 파싱
    # ─────────────────────────────────────────────────────────────────
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            yaml_block = parts[1]
            for line in yaml_block.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    key, val = key.strip(), val.strip().strip("'\"")
                    if key == "company" and val:
                        company = val
                    elif key == "doc_type" and val:
                        doc_type = val
                    elif key == "source" and val:
                        source = val
                    elif key == "date" and val:
                        date = val

    # ─────────────────────────────────────────────────────────────────
    # 4. 본문 정규식 패턴을 통한 출처(source) 및 발행일(date) 보완 추출
    # ─────────────────────────────────────────────────────────────────
    if not source and content:
        source_match = re.search(r"\*\s*\*\*(?:출처|발행기관|공식 표기\(Reference\))\*\*:\s*([^\n\r]+)", content)
        if source_match:
            source = source_match.group(1).strip()

    if not date and content:
        date_match = re.search(r"\*\s*\*\*(?:발간일|발행일|기준일)\*\*:\s*([^\n\r]+)", content)
        if date_match:
            date = date_match.group(1).strip()
        else:
            d_match = re.search(r"\((\d{4}[-.]\d{2}[-.]\d{2})\)", content[:1000])
            if d_match:
                date = d_match.group(1)

    # ─────────────────────────────────────────────────────────────────
    # 5. 기본값 Fallback
    # ─────────────────────────────────────────────────────────────────
    if not source:
        if "spri" in filename:
            source = "소프트웨어정책연구소 (SPRi)"
        elif "kiria" in filename:
            source = "한국로봇산업진흥원 (KIRIA)"
        elif company == "twinny":
            source = "(주)트위니"
        elif company == "seoulrobotics":
            source = "서울로보틱스"
        elif company == "neubility":
            source = "뉴빌리티"
        elif company == "xyz":
            source = "엑스와이지(XYZ)"
        elif company == "marsauto":
            source = "마스오토"
        else:
            source = os.path.basename(file_path)

    if not date:
        date = "2026-09-29"

    # 기업명 동의어/오타 정규화
    if company == "newbility":
        company = "neubility"
    elif company in ("seoul_robotics", "seoul robotics"):
        company = "seoulrobotics"

    return {
        "company": company,
        "doc_type": doc_type,
        "source": source,
        "date": date,
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
    }


def load_markdown_file(file_path: str) -> List[Document]:
    """
    마크다운(.md) 파일을 읽어 소제목(#, ##, ###) 단위로 1차 분할하고,
    1,500자를 초과하는 큰 섹션은 RecursiveCharacterTextSplitter로 재분할합니다.
    - 소제목 헤더 계층 정보가 Document.metadata에 유지됩니다.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    meta = extract_metadata(file_path, text)

    # 소제목 분할 기준 설정
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    try:
        header_splits = md_splitter.split_text(text)
    except Exception:
        header_splits = [Document(page_content=text, metadata={})]

    # 길이 초과 청크에 대한 재분할기
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    final_docs = []
    for split_doc in header_splits:
        # 파일 공통 메타데이터와 소제목 헤더 메타데이터 병합
        merged_meta = dict(meta)
        merged_meta.update(split_doc.metadata)

        sub_chunks = char_splitter.split_text(split_doc.page_content)
        for chunk in sub_chunks:
            if chunk.strip():
                final_docs.append(Document(page_content=chunk.strip(), metadata=merged_meta))

    return final_docs


def load_pdf_file(file_path: str) -> List[Document]:
    """
    PDF(.pdf) 파일을 로드하고 1,200자 단위로 청킹하여 Document 리스트를 반환합니다.
    """
    meta = extract_metadata(file_path, "")

    text_content = []
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for p in pages:
            text_content.append(p.page_content)
    except Exception:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text() or ""
            text_content.append(t)

    full_text = "\n\n".join(text_content)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = char_splitter.split_text(full_text)
    docs = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = i
            docs.append(Document(page_content=chunk.strip(), metadata=chunk_meta))
    return docs


def load_all_documents(data_dir: str = "data") -> List[Document]:
    """
    data/ 폴더 하위의 모든 .md 및 .pdf 파일을 탐색하여 청크 문서 리스트를 반환합니다.
    - 참조용 출처 목록 파일(_references*)이나 숨김 파일은 자동으로 제외됩니다.
    - [추가된 리포트] data/market/spri_physical_ai_report.md 도 자동으로 포함됩니다.
    """
    all_docs = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            # 참조용 목록 파일(_references.md 등) 및 숨김 파일 제외
            if f.startswith("_") or f.startswith("."):
                continue

            file_path = os.path.join(root, f)
            if f.endswith(".md"):
                docs = load_markdown_file(file_path)
                all_docs.extend(docs)
            elif f.endswith(".pdf"):
                docs = load_pdf_file(file_path)
                all_docs.extend(docs)

    return all_docs


def _wrap_filter(filter_arg: Any):
    """
    에이전트들이 전달하는 딕셔너리 필터(예: {'doc_type': ['tech', 'comprehensive']})를
    FAISS가 정상 인식할 수 있도록 콜러블(callable) 함수로 래핑합니다.
    """
    if filter_arg is None:
        return None
    if callable(filter_arg):
        return filter_arg
    if isinstance(filter_arg, dict):
        def _callable_filter(metadata: dict) -> bool:
            for key, expected in filter_arg.items():
                actual = metadata.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                else:
                    if actual != expected:
                        return False
            return True
        return _callable_filter
    return filter_arg


def _enhance_vectorstore(vectorstore: FAISS) -> FAISS:
    """
    FAISS 인스턴스의 similarity_search 및 similarity_search_with_score 메서드를 확장하여,
    딕셔너리 필터에 리스트가 포함되어 있어도 정상 동작하도록 지원합니다.
    """
    orig_similarity_search = vectorstore.similarity_search
    orig_similarity_search_with_score = vectorstore.similarity_search_with_score

    def enhanced_similarity_search(query: str, k: int = 4, filter: Any = None, **kwargs):
        wrapped = _wrap_filter(filter)
        return orig_similarity_search(query, k=k, filter=wrapped, **kwargs)

    def enhanced_similarity_search_with_score(query: str, k: int = 4, filter: Any = None, **kwargs):
        wrapped = _wrap_filter(filter)
        return orig_similarity_search_with_score(query, k=k, filter=wrapped, **kwargs)

    vectorstore.similarity_search = enhanced_similarity_search
    vectorstore.similarity_search_with_score = enhanced_similarity_search_with_score
    return vectorstore


def build_vectorstore(
    data_dir: str = "data",
    index_path: Optional[str] = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False,
) -> FAISS:
    """
    FAISS 벡터스토어를 생성하거나 디스크 캐시에서 불러옵니다. (모든 에이전트 공통 진입점)

    매개변수:
      - data_dir: 문서 원본이 저장된 폴더 (기본: 'data')
      - index_path: FAISS 인덱스 캐시 디렉토리 (기본: 'storage/faiss_index')
      - force_rebuild: True일 경우 캐시를 무시하고 문서 전체를 새로 임베딩

    반환값:
      - FAISS 벡터스토어 객체 (다중 조건 필터링 지원)

    사용 예시:
      >>> from src.vectorstore import build_vectorstore
      >>> vs = build_vectorstore()
      >>> docs = vs.similarity_search("자율주행 특허", k=3, filter={"company": "twinny", "doc_type": ["tech", "comprehensive"]})
    """
    embeddings = get_embeddings()

    # 디스크에 저장된 캐시 인덱스가 존재하면 즉시 로드 (수 초 내 완료)
    if index_path and not force_rebuild:
        faiss_file = os.path.join(index_path, "index.faiss")
        pkl_file = os.path.join(index_path, "index.pkl")
        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            print(f"[Vectorstore] 기존 FAISS 캐시 인덱스를 불러옵니다: '{index_path}'")
            try:
                vs = FAISS.load_local(
                    index_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                return _enhance_vectorstore(vs)
            except Exception as e:
                print(f"[Vectorstore] 캐시 로드 실패 ({e}), 문서를 새로 임베딩합니다...")

    # 문서 로드 및 청킹 실행
    print(f"[Vectorstore] '{data_dir}' 폴더에서 문서 로딩을 시작합니다...")
    documents = load_all_documents(data_dir)
    print(f"[Vectorstore] 총 {len(documents)}개의 청크가 생성되었습니다.")

    if not documents:
        raise ValueError(f"'{data_dir}' 폴더에서 유효한 문서를 찾을 수 없습니다.")

    print(f"[Vectorstore] '{EMBEDDING_MODEL_NAME}' 모델로 {len(documents)}개 청크를 임베딩합니다...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 디스크 캐시로 저장
    if index_path:
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        print(f"[Vectorstore] FAISS 인덱스를 '{index_path}'에 저장했습니다.")

    return _enhance_vectorstore(vectorstore)


def search_documents(
    vectorstore: FAISS,
    query: str,
    company: Optional[str] = None,
    doc_types: Optional[Union[str, List[str]]] = None,
    k: int = 5,
) -> List[Document]:
    """
    에이전트 개발자를 위한 간편 검색 도우미 함수:

    매개변수:
      - vectorstore: build_vectorstore()로 생성된 객체
      - query: 검색 쿼리 문자열 (예: "창업자 이력 및 학력")
      - company: 대상 스타트업 ('twinny', 'neubility', 'seoulrobotics', 'marsauto', 'xyz', 'common')
      - doc_types: 단일 문서유형 또는 리스트 (['tech', 'comprehensive'], 'market' 등)
      - k: 반환할 상위 문서 수 (기본: 5)
    """
    filters = {}
    if company is not None:
        filters["company"] = company
    if doc_types is not None:
        filters["doc_type"] = doc_types

    return vectorstore.similarity_search(query, k=k, filter=filters if filters else None)


def get_retriever(
    vectorstore: FAISS,
    company: Optional[str] = None,
    doc_types: Optional[Union[str, List[str]]] = None,
    k: int = 5,
):
    """
    특정 기업 및 문서 유형 필터가 사전 적용된 LangChain Retriever 객체를 반환합니다.
    LangGraph 에이전트 체인 구축 시 retriever로 바로 전달할 수 있습니다.
    """
    filters = {}
    if company is not None:
        filters["company"] = company
    if doc_types is not None:
        filters["doc_type"] = doc_types

    search_kwargs = {"k": k}
    if filters:
        search_kwargs["filter"] = _wrap_filter(filters)

    return vectorstore.as_retriever(search_kwargs=search_kwargs)
