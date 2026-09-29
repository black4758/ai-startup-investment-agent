# AI Startup Investment Evaluation Agent

본 프로젝트는 **Physical AI / Robotics** 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview
- **Objective** : Physical AI 스타트업의 기술력, 시장성, 경쟁력, 재무 실적 등을 기준으로 투자 적합성 분석
- **Method** : LangGraph 기반 Multi-Agent + Agentic RAG 파이프라인

## Tech Stack
- **Framework** : LangGraph, LangChain
- **LLM / Generator** : GPT-4o-mini
- **LLM / Judge** : GPT-4o-mini
- **Retrieval** : FAISS (In-Memory VectorStore)
- **Embedding** : BAAI/bge-m3 (Open-source Embedding)

## Directory Structure
```text
├── data/                  # 문서 풀 (스타트업 프로필 및 공통 시장 리포트)
├── agents/                # 평가 기준별 Agent 모듈
├── prompts/               # 프롬프트 템플릿
├── outputs/               # 평가 결과 저장
├── app.py                 # 실행 스크립트
└── README.md
```

## Features
- 비상장 Physical AI 스타트업 대상 자동 탐색 및 기본 정보 수집
- 기술 사양서 및 특허 기반 핵심 기술 요약 (RAG)
- 공공기관(KIRIA 등) 산업 리포트 기반 시장 규모 및 성장성 분석 (RAG)
- LLM 기반 국내외 경쟁사 비교 분석 및 차별성 도출
- VC 투자 심의 기준(Scorecard & Bessemer Checklist) 기반 투자/보류 판정
- 투자 판단 결과에 따른 조건부 루프(Loop) 및 최종 투자 심사 보고서 자동 생성

## Usage
```bash
python app.py
```
