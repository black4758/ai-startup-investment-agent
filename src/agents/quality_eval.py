"""
[src/agents/quality_eval.py]
보고서 품질 평가 노드 (Hybrid = 1안 규칙 검사 + 2안 LLM Judge, 항목별 거부권 방식)

흐름:
1. 규칙 검사 (결정적, API 호출 없음) : 4개 항목을 각각 코드로 검사
2. 규칙을 통과한 항목만 LLM Judge가 내용 채점 (1~5점, JUDGE_PASS_SCORE 이상 통과)
   → 항목 통과 = 규칙 통과 AND LLM 통과. 규칙 실패 항목은 LLM을 부르지 않는다 (비용 절감).
3. 하나라도 미달이면 항목별 사유를 feedback으로 담아 재작성 요청
   (재작성 최대 MAX_REVISIONS회, 그 뒤에도 미달이면 현재 보고서로 종료)

평가 항목 (노션 가이드 D. 보고서 품질 평가):
- Groundedness  : 본문 주장에 출처 번호[n]가 달려 있고, 그 번호가 실제 검색된 출처로 추적되는가
- 중립성        : 특정 기업/기술 추천, 우열·순위 판정이 없는가 (과제 정의 그대로)
- 편향 통제     : 본문 인용이 여러 출처에 분산되고, 기업별로 리스크·한계를 다루는가
- 관점 커버리지 : 기업별로 기술 성숙도·시장성·이해관계자·도메인 적용 4개 관점을 다루는가

보고서 형식 전제 (reporter 프롬프트에 반영 필요):
- 본문 서술 문장 끝에 [1], [2] 형식 출처 번호
- '## REFERENCE' 아래 '[n] 제목 - URL' 형식 목록 (URL이 없는 로컬 문서는 제목/파일명만)
- 기업별 섹션 제목에 기업명 포함

그래프 연결 (조정 계층에서 사용):
    workflow.add_node("quality_eval", evaluate_quality)
    workflow.add_conditional_edges("quality_eval", route_after_quality, {"reporter": ..., "end": ...})
    # QUALITY_FAIL_ROUTE=supervisor 로 쓰려면 매핑에 "supervisor"도 추가
"""

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.state import GraphState, QualityEvalResult

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "prompts" / "quality_eval_prompt.txt"
LOG_DIR = ROOT / "outputs" / "logs"

# ── 종료 보장: 품질 미달 시 재작성 최대 횟수 (state.report_revision_count 가드) ──
#    미달 1~MAX_REVISIONS회차 → 재작성, MAX_REVISIONS+1회차 미달 → 현재 보고서로 종료
MAX_REVISIONS = 3

# ── LLM Judge 통과 기준 (1~5점) ───────────────────────────────────────────
JUDGE_PASS_SCORE = 4

# ── Groundedness 규칙 ────────────────────────────────────────────────────
MIN_CITED_LINE_RATIO = 0.7   # 본문 서술 문장 중 출처 번호가 달린 비율
MIN_LINE_LEN = 20            # 이보다 짧은 줄(제목 조각, 라벨)은 서술 문장으로 보지 않음

# ── 편향 통제 규칙 ───────────────────────────────────────────────────────
MIN_CITED_SOURCES = 3        # 본문에서 실제 인용된 서로 다른 출처 수
MAX_SINGLE_SOURCE_SHARE = 0.4  # 단일 출처가 전체 인용에서 차지할 수 있는 최대 비중
MAX_DOMAIN_SHARE = 0.6       # (URL 있는 출처만) 한 도메인 최대 비중

# ── 중립성 규칙: 과제 정의 '특정 기술 추천/우열 판정 없음' 기준 ───────────────
#    '최고기술책임자', '점유율 1위(출처 인용)' 같은 사실 서술은 걸리지 않도록 표현 단위로 정의
BANNED_PATTERNS = [
    r"추천(?:합니다|한다|드립니다|드림|함)", r"추천\s*(?:기업|대상|순위)", r"강력\s*추천", r"투자\s*(?:를\s*)?권(?:유|장)",
    r"반드시\s*투자", r"최우수", r"압도적", r"독보적", r"유일무이", r"단연", r"우월",
    r"우위에\s*있", r"보다\s*(?:우수|뛰어나|낫)", r"가장\s*(?:유망|뛰어난|우수한|경쟁력\s*있는)",
    r"최고의\s*(?:기업|투자|선택|대안)", r"확실한\s*성공", r"의심의\s*여지", r"무조건",
]

# ── 관점 커버리지 규칙: 기업 섹션 안에 관점 소제목 또는 대표 키워드 ─────────────
#    '적용', '도입'처럼 어디에나 나오는 단어는 제외 (항상 통과하는 문제 방지)
PERSPECTIVE_KEYWORDS = {
    "기술 성숙도": ["기술 성숙도", "기술성숙도", "TRL", "특허", "상용화", "양산", "기술 차별"],
    "시장성": ["시장성", "시장 규모", "TAM", "SAM", "CAGR", "성장률", "경쟁 구도"],
    "이해관계자": ["이해관계자", "창업자", "창업팀", "경영진", "투자자", "투자사", "고객사", "파트너"],
    "도메인 적용": ["도메인 적용", "도입 사례", "적용 사례", "현장 적용", "도입 효과", "실증", "레퍼런스 고객"],
}
# 기업 섹션에 소제목이 있으면 소제목으로 판정 (키워드는 관점끼리 겹치므로 소제목이 없을 때만 사용)
PERSPECTIVE_HEADINGS = {
    "기술 성숙도": ["기술 성숙도", "기술성숙도", "기술성", "기술"],
    "시장성": ["시장"],
    "이해관계자": ["이해관계자", "창업", "팀", "투자자", "고객", "파트너"],
    "도메인 적용": ["도메인", "적용", "도입", "활용"],
}
RISK_KEYWORDS = ["리스크", "한계", "위험", "불확실", "정보 부재", "정보 공백", "우려"]

ITEMS = ("groundedness", "neutrality", "bias_control", "coverage")
ITEM_NAMES = {
    "groundedness": "Groundedness",
    "neutrality": "중립성",
    "bias_control": "편향 통제",
    "coverage": "관점 커버리지",
}
# 미달 시 원인: 글 문제(재작성) vs 자료 문제(재조사)
FIX_TARGET = {"groundedness": "reporter", "neutrality": "reporter",
              "bias_control": "supervisor", "coverage": "supervisor"}
# 자료 문제를 Supervisor 재조사로 보낼지 (그래프 매핑에 "supervisor"가 있을 때만 켤 것)
FAIL_ROUTE_SUPERVISOR = os.getenv("QUALITY_FAIL_ROUTE", "reporter") == "supervisor"


# ─────────────────────────────────────────────────────────────────────────────
# 0. LLM Judge 구조화 출력
# ─────────────────────────────────────────────────────────────────────────────

class JudgeItem(BaseModel):
    item: Literal["groundedness", "neutrality", "bias_control", "coverage"]
    score: int = Field(ge=1, le=5, description="1(심각한 미달) ~ 5(완전 충족)")
    reason: str = Field(description="채점 사유 1~2문장")
    problems: List[str] = Field(default_factory=list, description="문제 문장 발췌 (각 30자 이내)")


class LLMJudgeVerdict(BaseModel):
    """LLM Judge 구조화 출력 (규칙을 통과한 항목의 내용 판정 전용)"""
    items: List[JudgeItem]
    feedback: str = Field(default="", description="보고서 재작성을 위한 구체적인 수정 지침")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 공통 파서
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_HEADING_LINE_RE = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.*)$", re.MULTILINE)
_REF_HEADING_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*REFERENCES?\b.*$",
                             re.IGNORECASE | re.MULTILINE)
_SUMMARY_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(?:\*\*)?[ \t]*(?:\d+\.\s*)?SUMMARY\b",
                                 re.IGNORECASE | re.MULTILINE)


def extract_urls(text: str) -> List[str]:
    """문자열에서 URL을 순서대로 추출합니다. (끝의 문장부호·슬래시 제거, 비교용으로 소문자·http 통일)"""
    return [_norm_url(u) for u in _URL_RE.findall(text or "")]


def _norm_url(url: str) -> str:
    u = url.rstrip(".,;/").lower()
    u = re.sub(r"^http://", "https://", u)
    return re.sub(r"^https://www\.", "https://", u)


def _norm_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", (text or "").lower())


def split_reference(report: str) -> Tuple[str, Optional[str]]:
    """(본문, REFERENCE 섹션) 으로 나눕니다. REFERENCE가 없으면 (전체, None)."""
    m = _REF_HEADING_RE.search(report or "")
    if not m:
        return report or "", None
    return report[: m.start()], report[m.end():]


def parse_reference_entries(reference: str) -> Dict[int, str]:
    """'[n] 제목 - URL' 또는 'n. 제목 - URL' 형식의 REFERENCE 항목을 {번호: 내용}으로."""
    entries = {}
    for line in (reference or "").splitlines():
        m = re.match(r"^[ \t]*(?:[-*][ \t]*)?\[?(\d{1,3})[\].)][ \t]*(.+)$", line)
        if m:
            entries[int(m.group(1))] = m.group(2).strip()
    return entries


def narrative_lines(body: str) -> List[str]:
    """본문 서술 문장(불릿·문단)만 추립니다. 제목, 표, 짧은 라벨은 제외."""
    lines = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "|", ">", "```")) or re.fullmatch(r"[-*_=\s]+", line):
            continue
        text = re.sub(r"^[-*+][ \t]+|^\d+\.[ \t]+", "", line)
        if len(_CITE_RE.sub("", text).strip()) >= MIN_LINE_LEN:
            lines.append(line)
    return lines


def company_section(body: str, company: str) -> Optional[str]:
    """기업명이 들어간 제목 아래 섹션(같거나 상위 레벨 제목 전까지)을 반환."""
    headings = list(_HEADING_LINE_RE.finditer(body))
    for i, h in enumerate(headings):
        if company in h.group(2):
            level = len(h.group(1))
            end = len(body)
            for nxt in headings[i + 1:]:
                if len(nxt.group(1)) <= level:
                    end = nxt.start()
                    break
            return body[h.end(): end]
    return None


def _source_text(source: Any) -> str:
    if isinstance(source, dict):
        return " ".join(str(source.get(k, "")) for k in ("title", "url", "source", "reference", "file_name", "id"))
    return str(source)


def collect_sources(state: GraphState) -> List[str]:
    """
    하위 에이전트가 실제로 검색한 출처를 모읍니다. (제목·파일명·URL을 모두 대조에 사용)
    - state.sources / evaluations[*].sources / perspectives[*].sources
    """
    sources: List[Any] = list(state.get("sources") or [])
    if isinstance(state.get("sources"), dict):
        sources = list(state["sources"].values())
    for evaluation in state.get("evaluations") or []:
        sources.extend(evaluation.get("sources") or [])
    for perspective in (state.get("perspectives") or {}).values():
        if isinstance(perspective, dict):
            sources.extend(perspective.get("sources") or [])
        else:
            sources.extend(getattr(perspective, "sources", None) or [])
    return [s for s in dict.fromkeys(_source_text(s).strip() for s in sources) if s]


def collect_companies(state: GraphState) -> List[str]:
    """보고서가 다뤄야 할 기업 목록. (evaluations의 startup → companies/candidates 순으로 탐색)"""
    names = [ev.get("startup") for ev in state.get("evaluations") or [] if ev.get("startup")]
    if not names:
        names = list(state.get("companies") or [])
    return list(dict.fromkeys(names))


def _match_source(entry: str, sources: List[str]) -> bool:
    """REFERENCE 항목이 검색된 출처 중 하나와 대응하는지: URL 일치 또는 제목/파일명 포함."""
    entry_urls = set(extract_urls(entry))
    entry_text = _norm_text(_URL_RE.sub("", entry))
    for src in sources:
        if entry_urls and entry_urls & set(extract_urls(src)):
            return True
        src_text = _norm_text(_URL_RE.sub("", src))
        if len(src_text) >= 4 and len(entry_text) >= 4 and (src_text in entry_text or entry_text in src_text):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. 규칙 검사 (1안) - 순수 함수, API 키 없이 테스트 가능
# ─────────────────────────────────────────────────────────────────────────────

def check_groundedness(report: str, sources: List[str]) -> List[str]:
    """본문 서술의 출처 번호 비율 + 번호가 REFERENCE·검색 출처로 추적되는지 검사."""
    body, reference = split_reference(report)
    if reference is None:
        return ["REFERENCE 섹션이 없음"]

    violations = []
    entries = parse_reference_entries(reference)
    if not entries:
        violations.append("REFERENCE 항목이 '[n] 제목 - URL' 번호 형식이 아님 (본문 인용과 연결 불가)")
    if not sources:
        violations.append("검색된 출처(sources)가 비어 있어 주장을 추적할 수 없음")

    lines = narrative_lines(body)
    cited = [l for l in lines if _CITE_RE.search(l)]
    ratio = len(cited) / len(lines) if lines else 0.0
    if ratio < MIN_CITED_LINE_RATIO:
        sample = [l[:40] for l in lines if not _CITE_RE.search(l)][:3]
        violations.append(f"출처 번호가 달린 서술 {len(cited)}/{len(lines)}개({ratio:.0%}) < {MIN_CITED_LINE_RATIO:.0%}. 예: {sample}")

    if entries:
        dangling = sorted({int(n) for n in _CITE_RE.findall(body)} - set(entries))
        if dangling:
            violations.append(f"REFERENCE에 없는 출처 번호 인용: {dangling[:5]}")
    if entries and sources:
        unknown = [f"[{n}] {e[:40]}" for n, e in entries.items() if not _match_source(e, sources)]
        if unknown:
            violations.append(f"검색된 출처에서 찾을 수 없는 REFERENCE: {unknown[:3]}")
    return violations


def check_neutrality(report: str) -> List[str]:
    """추천·우열 판정 표현 검사. REFERENCE(기사 제목 등)는 작성자가 고칠 수 없으므로 제외."""
    body, _ = split_reference(report)
    hits = []
    for pat in BANNED_PATTERNS:
        for m in re.finditer(pat, body):
            hits.append(body[max(0, m.start() - 15): m.end() + 10].replace("\n", " "))
    return [f"추천·우열 판정 표현 {len(hits)}건: {hits[:4]}"] if hits else []


def check_bias_control(report: str, companies: List[str]) -> List[str]:
    """본문 인용의 출처 분산 + 기업별 리스크·한계 서술 검사."""
    body, reference = split_reference(report)
    violations = []

    counts = Counter(_CITE_RE.findall(body))
    total = sum(counts.values())
    n_refs = len(parse_reference_entries(reference or ""))
    need = min(MIN_CITED_SOURCES, n_refs) if n_refs else MIN_CITED_SOURCES
    if len(counts) < need:
        violations.append(f"본문에서 인용된 서로 다른 출처 {len(counts)}개 < {need}개 (단일 출처 의존)")
    if total and len(counts) > 1:
        top, n = counts.most_common(1)[0]
        if n / total > MAX_SINGLE_SOURCE_SHARE:
            violations.append(f"출처 [{top}]가 전체 인용의 {n / total:.0%} 차지 (> {MAX_SINGLE_SOURCE_SHARE:.0%})")

    urls = list(dict.fromkeys(extract_urls(reference or "")))
    domains = [urlparse(u).netloc for u in urls]
    if len(domains) >= 3:
        top_domain, n = Counter(domains).most_common(1)[0]
        if n / len(domains) > MAX_DOMAIN_SHARE:
            violations.append(f"REFERENCE 링크의 {n / len(domains):.0%}가 {top_domain} 한 곳에 편중")

    targets = [(c, company_section(body, c)) for c in companies] or [("보고서 전체", body)]
    no_risk = [c for c, sec in targets if sec is not None and not any(k in sec for k in RISK_KEYWORDS)]
    if no_risk:
        violations.append(f"리스크·한계 서술이 없는 기업: {no_risk} (유리한 근거만 제시)")
    return violations


def check_coverage(report: str, companies: List[str]) -> List[str]:
    """SUMMARY 존재 + 기업별 섹션 안에 4개 관점이 모두 있는지 검사."""
    body, _ = split_reference(report)
    violations = []
    if not _SUMMARY_HEADING_RE.search(body):
        violations.append("SUMMARY 섹션이 없음")

    targets = [(c, company_section(body, c)) for c in companies] or [("보고서 전체", body)]
    for name, sec in targets:
        if sec is None:
            violations.append(f"{name} 섹션(기업명이 들어간 제목)이 없음")
            continue
        subheads = [h.group(2) for h in _HEADING_LINE_RE.finditer(sec)]
        if subheads:
            missing = [p for p, alias in PERSPECTIVE_HEADINGS.items()
                       if not any(a in h for h in subheads for a in alias)]
        else:
            missing = [p for p, kws in PERSPECTIVE_KEYWORDS.items() if not any(k in sec for k in kws)]
        if missing:
            violations.append(f"{name}: 누락된 관점 {missing}")
    return violations


def run_rule_checks(report: str, sources: List[str], companies: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """4개 항목 규칙 검사 결과를 {항목: [미달 사유]}로 반환합니다. (빈 리스트 = 통과)"""
    companies = companies or []
    return {
        "groundedness": check_groundedness(report, sources),
        "neutrality": check_neutrality(report),
        "bias_control": check_bias_control(report, companies),
        "coverage": check_coverage(report, companies),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM Judge (2안)
# ─────────────────────────────────────────────────────────────────────────────

def get_judge_llm():
    """Judge 전용 LLM. 자기 채점 편향을 줄이려면 OPENAI_JUDGE_MODEL을 생성 모델과 다르게 지정."""
    from langchain_openai import ChatOpenAI

    gen_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model_name = os.getenv("OPENAI_JUDGE_MODEL") or gen_model
    if model_name == gen_model:
        logger.warning("OPENAI_JUDGE_MODEL이 생성 모델과 같음(%s) — 자기 채점 편향 가능", model_name)
    return ChatOpenAI(model=model_name, temperature=0.0, max_retries=2)


def _compact_evaluations(state: GraphState) -> List[Dict[str, Any]]:
    """근거 대조에 필요한 필드만 추립니다. (점수·판정·분석 텍스트, 길이 제한)"""
    out = []
    for ev in state.get("evaluations") or []:
        row = {}
        for key, val in ev.items():
            if key == "startup" or key.endswith(("_analysis", "_scores", "scores")) or key in ("total_score", "decision"):
                row[key] = val[:1500] if isinstance(val, str) else val
        out.append(row)
    return out


def _build_prompt(state: GraphState, items: List[str]) -> str:
    """.format()은 JSON 문자열의 중괄호 때문에 깨지므로 replace로 치환한다."""
    values = {
        "{items}": "\n".join(f"- {i} ({ITEM_NAMES[i]})" for i in items),
        "{sources}": "\n".join(f"- {s}" for s in collect_sources(state)) or "정보 없음",
        "{evaluations}": json.dumps(_compact_evaluations(state), ensure_ascii=False, indent=1, default=str),
        "{report}": state.get("report", ""),
    }
    template = PROMPT_PATH.read_text(encoding="utf-8")
    if "{items}" not in template:   # 예전 프롬프트 파일이어도 채점 대상은 전달
        template = "[채점 대상 항목]\n{items}\n\n" + template
    for placeholder, value in values.items():
        template = template.replace(placeholder, value)
    return template


def run_llm_judge(state: GraphState, items: List[str]) -> LLMJudgeVerdict:
    llm = get_judge_llm().with_structured_output(LLMJudgeVerdict)
    return llm.invoke(_build_prompt(state, items))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid 결합 · LangGraph 노드 · 라우팅
# ─────────────────────────────────────────────────────────────────────────────

def _to_result(passed: Dict[str, bool], feedback: str) -> QualityEvalResult:
    """
    4개 항목 판정을 State 스키마(QualityEvalResult)로 옮깁니다.
    State 모델에 있는 필드만 채우므로 coverage_pass 필드를 추가하면 자동으로 저장됩니다.
    """
    verdict = "PASS" if all(passed.values()) else "FAIL"
    values = {
        "groundedness_pass": passed["groundedness"],
        "neutrality_pass": passed["neutrality"],
        "bias_control_pass": passed["bias_control"],
        "coverage_pass": passed["coverage"],
        "verdict": verdict,
        "feedback": "" if verdict == "PASS" else feedback,
    }
    fields = getattr(QualityEvalResult, "model_fields", None)
    if fields is not None:
        if "coverage_pass" not in fields:
            logger.warning("QualityEvalResult에 coverage_pass 필드가 없음 — state.py에 추가 필요")
        values = {k: v for k, v in values.items() if k in fields}
    return QualityEvalResult(**values)


def judge_report(state: GraphState) -> Tuple[QualityEvalResult, Dict[str, Any]]:
    """
    항목별 거부권 방식 Hybrid 판정.
      항목 통과 = 규칙 통과 AND LLM 점수 >= JUDGE_PASS_SCORE
    반환: (State에 저장할 판정, 로그용 상세 {items, reasons, fix_targets, error})
    """
    report = state.get("report", "")
    rule = run_rule_checks(report, collect_sources(state), collect_companies(state))
    to_judge = [i for i in ITEMS if not rule[i]]

    judged: Dict[str, JudgeItem] = {}
    llm_feedback, error = "", None
    if to_judge:
        try:
            verdict = run_llm_judge(state, to_judge)
            judged = {j.item: j for j in verdict.items if j.item in to_judge}
            llm_feedback = verdict.feedback
        except Exception as e:
            # Fail-closed: Judge 실패 시 내용 검증이 안 된 것이므로 통과시키지 않음 (재작성 상한이 루프를 끊음)
            error = f"quality_eval LLM Judge 실패: {type(e).__name__}: {e}"

    passed, reasons, items_log = {}, [], {}
    for item in ITEMS:
        name = ITEM_NAMES[item]
        if rule[item]:
            ok, why = False, [f"{name} [규칙] {r}" for r in rule[item]]
            items_log[item] = {"rule": False, "llm": None}
        elif item not in judged:
            ok, why = False, [f"{name} [LLM] " + ("Judge 오류로 내용 검증 불가" if error else "판정 누락")]
            items_log[item] = {"rule": True, "llm": None}
        else:
            j = judged[item]
            ok = j.score >= JUDGE_PASS_SCORE
            why = [] if ok else [f"{name} [LLM {j.score}/5] {j.reason}" + (f" 문제: {j.problems[:3]}" if j.problems else "")]
            items_log[item] = {"rule": True, "llm": j.score}
        passed[item] = ok
        reasons += why

    failed = [i for i in ITEMS if not passed[i]]
    feedback = "\n".join(
        ["다음 항목이 품질 기준에 미달했습니다. 사유를 반영해 보고서를 수정하십시오."]
        + [f"- {r}" for r in reasons]
        + ([f"- 수정 지침: {llm_feedback}"] if llm_feedback else [])
    ) if failed else ""
    detail = {
        "items": items_log,
        "passed": passed,
        "reasons": reasons,
        "fix_targets": sorted({FIX_TARGET[i] for i in failed}),
        "error": error,
    }
    return _to_result(passed, feedback), detail


def _log_decision(record: Dict[str, Any]) -> None:
    """결정 로그는 State가 아니라 외부 파일로 적재 (trace_id로 State·LangSmith와 상관)."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{record.get('trace_id') or 'no_trace'}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.warning("결정 로그 파일 기록 실패: %s", e)
    logger.info(line)


def evaluate_quality(state: GraphState) -> Dict[str, Any]:
    """
    보고서 품질 평가 노드:
    eval_result에 판정을 기록하고, 미달이면 report_revision_count를 1 올립니다.
    """
    result, detail = judge_report(state)
    revision = state.get("report_revision_count", 0)
    if result.verdict == "FAIL":
        revision += 1

    _log_decision({
        "trace_id": state.get("trace_id", ""),
        "node": "quality_eval",
        "decision": result.verdict,
        "items": detail["items"],
        "fix_targets": detail["fix_targets"],
        "revision": revision,
        "reason": detail["reasons"],
        "error": detail["error"],
    })

    print(f"[품질 평가] 판정 {result.verdict} / 미달 누적 {revision}회 (재작성 상한 {MAX_REVISIONS}회)")
    for item, log in detail["items"].items():
        llm = "-" if log["llm"] is None else f"{log['llm']}/5"
        print(f"  {ITEM_NAMES[item]:<10} 규칙 {'O' if log['rule'] else 'X'}  LLM {llm}  → {'통과' if detail['passed'][item] else '미달'}")
    for reason in detail["reasons"]:
        print(f"  - {reason}")

    return {
        "eval_result": result,
        "report_revision_count": revision,
        "step_count": state.get("step_count", 0) + 1,
        "node_status": {**(state.get("node_status") or {}), "quality_eval": result.verdict},
        "last_error": detail["error"],
    }


def route_after_quality(state: GraphState) -> str:
    """
    품질 평가 후 분기:
    - PASS                                   -> 'end'
    - FAIL이고 미달 횟수가 MAX_REVISIONS 초과 -> 'end' (현재 보고서로 종료, 종료 보장)
    - FAIL이고 재작성 여유 있음              -> 'reporter' (피드백 반영 재작성)
      · QUALITY_FAIL_ROUTE=supervisor 이고 미달이 자료 문제(편향·커버리지)뿐이면 -> 'supervisor' (재조사)
    """
    result = state.get("eval_result")
    if result is None or result.verdict == "PASS":
        return "end"
    if state.get("report_revision_count", 0) > MAX_REVISIONS:
        print(f"[품질 평가] 재작성 상한({MAX_REVISIONS}회) 도달, 현재 보고서로 종료")
        return "end"
    if FAIL_ROUTE_SUPERVISOR:
        writing_ok = getattr(result, "groundedness_pass", True) and getattr(result, "neutrality_pass", True)
        if writing_ok:
            return "supervisor"
    return "reporter"
