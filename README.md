# llm_fin_audit

A hybrid AI system that measures when LLM financial agents hallucinate numeric values, violate hard constraints, or break formal rules — and corrects those failures using classical AI techniques.

## CS 664 Term Project
BU MET — Artificial Intelligence (Spring 2026)
Student: Lucky Upadhyay

## What it does
Compares three conditions on financial analysis tasks:
- LLM-only agent
- Classical AI solver only  
- Hybrid (LLM proposes, classical verifies)

## Three Core Tasks
1. Financial ratio constraint checking
2. Earnings anomaly detection
3. Compliance rule verification

## Setup
```bash
conda activate spring_2026
pip install -r requirements.txt
```

## Run
```bash
python src/evaluation/benchmark.py
```
