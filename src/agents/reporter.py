import os
import json
from datetime import datetime
from markdown_pdf import Section, MarkdownPdf
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def report_agent(state: dict) -> dict:
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
    
    # 3. LLM 초기화 (프로젝트 환경에 맞춰 모델명 변경 가능)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    # 출처 중복 제거 및 문자열 변환
    unique_sources = "\n".join(dict.fromkeys(sources)) if sources else "정보 없음"
    
    # 4. 보고서 생성 실행
    response = chain.invoke({
        "evaluations": json.dumps(evaluations, ensure_ascii=False, indent=2),
        "tech_analysis": tech_analysis,
        "company_analysis": company_analysis,
        "market_analysis": market_analysis,
        "sources": unique_sources
    })
    
    report_content = response.content

    # 5. 지정된 경로에 PDF 파일로 저장
    os.makedirs("outputs", exist_ok=True)
    
    # 팀 제출 규격에 맞춘 파일명[cite: 4]
    file_name = "RAG-Output_울산-2반_우성윤+이진서+김지훈+김영제+서승현.pdf"
    pdf_file_path = os.path.join("outputs", file_name)
    
    # 마크다운 텍스트를 PDF로 변환
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(report_content))
    pdf.save(pdf_file_path)

    # 6. 다음 노드를 위해 State 업데이트 반환
    return {"report": report_content}