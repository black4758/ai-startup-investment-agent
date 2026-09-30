# -*- coding: utf-8 -*-
"""
시장성 평가 에이전트 (3번 노드)
담당: 김지훈
평가 항목: 시장 내 포지션 및 진입장벽(15%), 시장성(10%)

설계 원칙
- 이 에이전트만 company 필터를 쓰지 않는다. 시장 자료는 5개사가 공유하는
  공통 문서이므로 doc_type == "market" 으로만 거른다.
- 추출은 모델이, 검증은 코드가 한다. 인용문이 실제 문서에 있는지 대조하고,
  정보 부재 항목은 코드가 2점으로 강제한다.
- 파싱에 실패해도 그래프가 멈추지 않는다. 2점 처리하고 넘어간다.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from src.state import GraphState
from src.vectorstore import build_vectorstore, search_documents

_VECTORSTORE = None

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "market_prompt.txt"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("MARKET_TOP_K", "6"))

# 프롬프트가 한글 라벨로 출력하므로 State 키로 변환한다
KEY_MAP = {
    "시장내포지션및진입장벽": "market_position_and_barriers",
    "시장성": "market_potential",
}
DEFAULT_SCORE = 2.0       # 정보 부재 시 기본값
CLAIM_CAP = 4.0           # 회사 자체 주장 수치만 있을 때 상한


def _get_vectorstore():
    """FAISS 인덱스는 로딩 비용이 크므로 한 번만 만들어 재사용한다."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = build_vectorstore()
    return _VECTORSTORE


def retrieve_market_docs(vectorstore, startup: str, k: int = TOP_K):
    """시장 문서만 검색한다. company 필터를 걸면 0건이 나온다."""
    query = (f"{startup} 시장 규모 TAM 연평균 성장률 CAGR 전망 "
             "규제 정책 진입 장벽 경쟁 구도")
    return search_documents(vectorstore, query, doc_types="market", k=k)


def format_docs(docs) -> str:
    """문서마다 인용용 태그를 완성된 형태로 붙인다."""
    if not docs:
        return "(검색된 시장 문서가 없습니다)"
    blocks = []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source") or "제목 미상"
        blocks.append(f"[문서: {title}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _section(text: str, name: str) -> str:
    m = re.search(rf"\[{name}\]\s*(.*?)(?=\n\s*\[[가-힣A-Za-z]+\]|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def parse_response(text: str) -> dict:
    """모델 출력을 dict로 바꾼다. 형식이 깨져도 예외를 던지지 않는다."""
    scores = {}
    block = _section(text, "점수")
    for label, key in KEY_MAP.items():
        m = re.search(rf"{label}\s*[:：]\s*(\d)", block)
        if m:
            scores[key] = float(max(1, min(5, int(m.group(1)))))

    raw = _section(text, "정보부재항목")
    missing = []
    if raw and "없음" not in raw:
        missing = [s.strip(" -•\t") for s in re.split(r"[,\n]", raw) if s.strip(" -•\t")]

    return {"analysis": _section(text, "분석") or text.strip(),
            "scores": scores, "missing": missing}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def verify_quotes(analysis: str, docs) -> tuple[list, list]:
    """분석문의 큰따옴표 인용이 실제 문서에 있는지 대조한다.
    LLM이 그럴듯한 시장 수치를 만들어내는 것을 코드가 걸러낸다."""
    corpus = _norm("\n".join(d.page_content for d in docs))
    quotes = re.findall(r"[\"“]([^\"”]{5,80})[\"”]", analysis)
    ok, bad = [], []
    for q in quotes:
        (ok if _norm(q)[:20] in corpus else bad).append(q)
    return ok, bad


def collect_sources(docs) -> list[str]:
    """출처는 모델이 쓴 것이 아니라 실제 검색된 문서에서 코드가 만든다.
    모델이 출처 제목을 지어내면 보고서 REFERENCE가 오염되기 때문이다.

    보고서 생성 에이전트가 참고문헌 링크를 만들 수 있도록 URL을 함께 담는다.
    형식: "문서명 (URL)" — URL이 없으면 문서명만.
    """
    seen, out = set(), []
    for d in docs:
        title = d.metadata.get("title") or d.metadata.get("source")
        if not title:
            continue
        url = d.metadata.get("url")
        label = f"{title} ({url})" if url else str(title)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def validate(parsed: dict, docs) -> tuple[dict, list, list]:
    """점수를 규칙에 맞게 보정하고 사유를 남긴다."""
    scores = dict(parsed["scores"])
    missing = list(parsed["missing"])
    notes = []

    if not docs:
        return ({k: DEFAULT_SCORE for k in KEY_MAP.values()},
                ["시장 문서 검색 결과 없음"],
                ["검색 결과 0건 → 두 항목 모두 2점 처리"])

    for key in KEY_MAP.values():
        if key not in scores:
            scores[key] = DEFAULT_SCORE
            missing.append(f"{key} 점수 파싱 실패")
            notes.append(f"{key} 점수를 읽지 못해 2점 처리")

    ok, bad = verify_quotes(parsed["analysis"], docs)
    if bad:
        notes.append(f"문서에 없는 인용 {len(bad)}건")
        for key in KEY_MAP.values():
            if scores[key] > CLAIM_CAP:
                scores[key] = CLAIM_CAP
                notes.append(f"{key}: 미검증 인용 → 최대 4점 제한")
    if not ok:
        missing.append("원문 인용 근거 없음")

    joined = " ".join(missing)
    if any(w in joined for w in ("TAM", "시장 규모", "CAGR", "성장률")):
        if scores["market_potential"] > DEFAULT_SCORE:
            scores["market_potential"] = DEFAULT_SCORE
            notes.append("TAM 또는 CAGR 정보 부재 → 시장성 2점 처리")

    corpus = "\n".join(d.page_content for d in docs)
    if not re.search(r"CAGR|연평균\s*성장", corpus) and scores["market_potential"] >= 5:
        scores["market_potential"] = 4.0
        notes.append("문서에 CAGR 근거 없음 → 시장성 5점 → 4점 하향")

    return scores, missing, notes


def evaluate_market(state: GraphState) -> Dict[str, Any]:
    """김지훈 담당: 시장성 평가 에이전트"""
    startup = state.get("current_startup") or state.get("company") or ""

    vectorstore = _get_vectorstore()
    docs = retrieve_market_docs(vectorstore, startup)

    prompt = (PROMPT_PATH.read_text(encoding="utf-8")
              .replace("{startup}", startup)
              .replace("{context}", format_docs(docs)))

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_retries=2)
    raw = llm.invoke(prompt).content

    parsed = parse_response(raw)
    scores, missing, notes = validate(parsed, docs)

    analysis = parsed["analysis"]
    if notes:
        analysis += "\n\n[코드 보정] " + " / ".join(notes)

    return {
        "market_analysis": analysis,
        "market_scores": scores,
        "missing_items": missing,
        "sources": collect_sources(docs),
    }
