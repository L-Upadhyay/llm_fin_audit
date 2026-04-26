# llm_fin_audit — CS 664 Term Project

## Project Overview
This project measures when LLM financial agents hallucinate numeric values, violate hard constraints, or break formal rules — and then corrects those failures using classical AI techniques.

## Student
Lucky Upadhyay — BU MET MS Applied Data Analytics
Course: MET CS 664 Artificial Intelligence (Prof. Suresh Kalathur, Spring 2026)

## Architecture
Two cooperating layers:
- LLM layer (Agno + Ollama): proposes financial analysis
- Classical layer (CSP + KB): verifies and corrects
The LLM cannot output a final verdict until the classical layer verifies it first.

## Three Core Tasks
1. Financial ratio constraint checking (CSP)
2. Earnings anomaly detection (search)
3. Compliance rule verification (KB + forward chaining)

## Classical Components (write from scratch, standard library only)
- CSP solver: backtracking + forward checking + AC-3
- Knowledge base: Clause/KB classes + forward chaining
- All subclassing Kalathur's AIMA Agent/Environment/Problem hierarchy

## LLM Components
- Agno agents running on local Ollama (llama3.2, nomic-embed-text)
- Custom @tool functions that call the classical layer
- Multi-agent team: DataAgent,  AnalysisAgent, ComplianceAgent, ReportAgent