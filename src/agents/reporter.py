import os
import json
from datetime import datetime
from markdown_pdf import Section, MarkdownPdf
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def generate_report(state: dict) -> dict:
    # 1. State에서 필요한 데이터 추출
    evaluations = state.get("evaluations", [])
    tech_analysis = state.get("tech_analysis", "")
    company_analysis = state.get("company_analysis", "")
    market_analysis = state.get("market_analysis", "")
    sources = state.get("sources", [])

    # 2. 프롬프트 로드
    with open("prompts/report_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = PromptTemplate.from_template(prompt_template)
    
    # 3. LLM 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    # 출처 중복 제거 및 리스트/링크 형태로 정돈
    if sources:
        cleaned_sources = []
        for s in dict.fromkeys(sources):
            if isinstance(s, dict):
                title = s.get("title", "참고 문서")
                url = s.get("url", "#")
                cleaned_sources.append(f"- [{title}]({url})")
            else:
                cleaned_sources.append(f"- {s}")
        unique_sources = "\n".join(cleaned_sources)
    else:
        unique_sources = "정보 없음"
    
    # 4. 보고서 생성 실행
    response = chain.invoke({
        "evaluations": json.dumps(evaluations, ensure_ascii=False, indent=2),
        "tech_analysis": tech_analysis,
        "company_analysis": company_analysis,
        "market_analysis": market_analysis,
        "sources": unique_sources
    })
    
    report_content = response.content

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
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(report_content))
    pdf.save(pdf_file_path)

    # 6. 다음 노드를 위해 State 업데이트 반환
    return {"report": report_content, "md_file_path": md_file_path, "pdf_file_path": pdf_file_path}

# 하위 호환성 별칭
report_agent = generate_report