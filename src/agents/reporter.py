import os
import json
from datetime import datetime
from markdown_pdf import Section, MarkdownPdf
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def generate_report(state: dict) -> dict:
      # 1. State에서 필요한 데이터 추출
    evaluations = state.get("evaluations", [])

    # 기업별 분석 결과 통합
    tech_analysis = []
    company_analysis = []
    market_analysis = []
    sources = []

    for ev in evaluations:
        company_name = (
            ev.get("company")
            or ev.get("company_name")
            or ev.get("startup")
            or "기업명 미확인"
        )

        if ev.get("tech_analysis"):
            tech_analysis.append(
                f"[{company_name}]\n{ev['tech_analysis']}"
            )

        if ev.get("company_analysis"):
            company_analysis.append(
                f"[{company_name}]\n{ev['company_analysis']}"
            )

        if ev.get("market_analysis"):
            market_analysis.append(
                f"[{company_name}]\n{ev['market_analysis']}"
            )

        sources.extend(ev.get("sources") or [])

    tech_analysis = "\n\n".join(tech_analysis)
    company_analysis = "\n\n".join(company_analysis)
    market_analysis = "\n\n".join(market_analysis)


    # 1-1. 품질 평가 결과 및 이전 보고서 추출
    eval_result = state.get("eval_result")
    previous_report = state.get("report", "")
    quality_feedback = ""

    if eval_result is not None:
        if isinstance(eval_result, dict):
            verdict = eval_result.get("verdict", "")
            feedback = eval_result.get("feedback", "")
        else:
            verdict = eval_result.verdict
            feedback = eval_result.feedback

        if verdict == "FAIL":
            quality_feedback = feedback

    # 최초 생성 시에는 이전 보고서를 사용하지 않음
    if not quality_feedback:
        previous_report = ""



    # 2. 프롬프트 로드
    with open("prompts/report_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = PromptTemplate.from_template(prompt_template)
    
    # 3. LLM 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
           # 출처 중복 제거: 동일 제목은 URL이 있는 항목 우선
    import re

    cleaned_sources = []
    source_index = {}

    for s in sources:
        if isinstance(s, dict):
            title = str(s.get("title") or "참고 문서").strip()
            url = str(s.get("url") or "").strip()
        else:
            raw = str(s).strip()

            # [제목](URL) 형식
            match = re.fullmatch(r"\[(.*?)\]\((https?://.*?)\)", raw)

            if match:
                title, url = match.groups()
            else:
                # 제목. URL 형식도 처리
                match = re.fullmatch(r"(.*?)\s+(https?://\S+)", raw)
                if match:
                    title, url = match.groups()
                else:
                    title, url = raw, ""

        title = title.strip()
        url = url.strip()

        # 제목이 동일한 경우 중복으로 판단
        key = re.sub(r"\s+", " ", title).strip().rstrip(".。").casefold()
        if not key:
            continue

        formatted = f"[{title}]({url})" if url else title

        if key in source_index:
            idx = source_index[key]

            # 기존 항목에 URL이 없고 새 항목에 URL이 있으면 교체
            if url and not cleaned_sources[idx][1]:
                cleaned_sources[idx] = (formatted, url)
        else:
            source_index[key] = len(cleaned_sources)
            cleaned_sources.append((formatted, url))

    numbered_sources = [
        f"[{i}] {formatted}"
        for i, (formatted, url) in enumerate(cleaned_sources, start=1)
    ]

    unique_sources = "\n".join(numbered_sources) or "정보 없음"


    
    # 4. 보고서 생성 실행
    print("[DEBUG] previous_report 길이:", len(previous_report))
    print("[DEBUG] quality_feedback:", quality_feedback)
    print("[DEBUG] 평가 기업 수:", len(evaluations))
    print("[DEBUG] 기술성 분석 길이:", len(tech_analysis))
    print("[DEBUG] 기업 분석 길이:", len(company_analysis))
    print("[DEBUG] 시장성 분석 길이:", len(market_analysis))
    print("[DEBUG] 수집된 출처 수:", len(sources))
    response = chain.invoke({
        "evaluations": json.dumps(evaluations, ensure_ascii=False, indent=2),
        "tech_analysis": tech_analysis,
        "company_analysis": company_analysis,
        "market_analysis": market_analysis,
        "sources": unique_sources,
        "previous_report": previous_report,      # 추가
        "quality_feedback": quality_feedback     # 추가
    })
    
    report_content = response.content
    # 보고서 최상단 제목 자동 추가
    report_content = report_content.strip()

    if not report_content.startswith("# 투자 검토 보고서"):
        report_content = "# 투자 검토 보고서\n\n" + report_content

    # LLM이 REFERENCE를 작성했더라도 제거
    import re

    report_content = re.split(
        r"(?im)^#{1,6}\s+REFERENCE(?:S)?\s*$",
        report_content,
        maxsplit=1
    )[0].rstrip()

    # 코드에서 REFERENCE 생성
   # 출처 번호별로 독립된 문단 생성
    reference_items = re.split(
    r"\s+(?=\[\d+\]\s)",
    unique_sources.strip()
)

    formatted_references = "\n\n".join(
    item.strip() for item in reference_items if item.strip()
)

    report_content += "\n\n# REFERENCE\n\n" + formatted_references + "\n"
    # 5. outputs 폴더 및 파일 경로 설정 (타임스탬프 적용)
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"RAG-Output_울산-2반_우성윤+이진서+김지훈+김영제+서승현_{timestamp}"
    
    # 5-1. Markdown 파일(.md) 저장
    md_file_path = os.path.join("outputs", f"{base_name}.md")
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

        # 5-2. PDF 파일(.pdf) 저장
    pdf_file_path = os.path.join("outputs", f"{base_name}.pdf")

    import re

    # PDF 변환 시 목적지가 없는 링크 제거
    pdf_content = re.sub(r'\[([^\]]+)\]\(#\)', r'\1', report_content)

    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(pdf_content, paper_size="A4"))
    pdf.save(pdf_file_path)
    import pymupdf

    with pymupdf.open(pdf_file_path) as doc:
        page_count = len(doc)
    print(f"[REPORTER] PDF 페이지 수: {page_count}/10")

    if page_count > 10:
        print("[WARNING] 보고서가 10페이지를 초과했습니다.")
    # 6. 다음 노드를 위해 State 업데이트 반환
    return {"report": report_content}


# 하위 호환성 별칭
report_agent = generate_report