# FailProof-LLM: Robustness and Domain-Specific Benchmarking for Large Language Models

## Overview

Large Language Models (LLMs) are powerful but remain vulnerable to **adversarial prompts**, **formatting stress tests**, and **domain-specific edge cases**.  
These vulnerabilities can cause models to:

- Produce incorrect or unsafe outputs
- Fail to comply with structured output requirements
- Break under edge-case or adversarial inputs

There is currently no **practical, end-to-end benchmark** for evaluating models across both **adversarial robustness** and **domain accuracy**.  
**FailProof-LLM** addresses this gap by providing a unified, extensible platform for **dataset generation, model evaluation, automated repair, and interactive analytics**.

---

## Key Features

### 1. Synthetic Dataset Generation
- Prompt-driven creation of high-quality test cases.
- Coverage across multiple robustness categories and real-world domains.

### 2. Multi-Model Evaluation
- Side-by-side comparison of multiple LLMs.
- Evaluation across structured, adversarial, and domain-specific tasks.

### 3. Automated Validation & Repair
- Validation using JSON schema checks, CSV header verification, HTML parsing, and more.
- Automatic repair attempts for failed outputs.

### 4. Interactive Analytics
- Real-time benchmarking dashboard.
- Domain/category breakdowns with model-level comparisons.

---

## Usage Instructions

1. **Home Page**  
   Select your workflow: *Dataset Studio* or *Run & Analyze*.

2. **Generate Datasets**  
   In *Dataset Studio*, select a category/domain, choose the dataset size, and generate test cases.

3. **Run Evaluation**  
   In *Run & Analyze*, select your dataset and choose models to evaluate.

4. **View Results**  
   Explore validation pass/fail breakdowns, repair success rates, latency metrics, and comparative model performance.

5. **Benchmark**  
   Review aggregated metrics per category/domain to identify model strengths and weaknesses.

---

## Models Considered

The platform supports both proprietary and open-source models. For initial experiments, the following were evaluated:

- **GPT-4o-mini** (OpenAI) — High accuracy, strong structured output robustness.
- **Qwen1.5-0.5B** (Alibaba) — Mid-range performance.
- **TinyLlama-1.1B-Chat** — Lightweight, low-compute option.
- **FLAN-T5-base** and **FLAN-T5-small** (Google) — Open-source baselines.

---

## Domains and Adversarial Categories

Evaluation is performed across **three primary domains** and **22 adversarial robustness categories**.

### Domains
- Financial
- Healthcare
- Legal

### Adversarial Categories

1. **Prompt Manipulation Attacks**
   - Prompt Injection  
   - Code Injection  
   - HTML Injection  

2. **Structured Output Stress Tests**
   - JSON Schema Stress  
   - CSV Formatting Stress  

3. **Cognitive & Reasoning Challenges**
   - Context Length Stress  
   - Multi-Step Reasoning Chains  
   - Numerical Reasoning Edge Cases  
   - Unit Conversion Traps  

4. **Language & Input Perturbations**
   - Unicode Confusables  
   - Homoglyph Attacks  
   - Mixed-Language Code Switch  
   - Low-Resource Language Prompts  

5. **Security & Safety Red-Teaming**
   - Red-Teaming Scenarios  
   - Chain-of-Thought Leakage  
   - Few-Shot Poisoning  

6. **Context & Misdirection**
   - Ambiguous Queries  
   - Context Switching  
   - Fake Context Misdirection  
   - Obfuscated Instructions  

7. **Domain-Specific Edge Cases**
   - Domain-specific scenarios in finance, law, and healthcare.

---

## Evaluation Metrics

Performance is measured across several dimensions:

- **Accuracy** — Percentage of cases passing validation.
- **Validation Pass Rate** — Domain-specific validator success rate.
- **Repair Success Rate** — Percentage of failed cases fixed by auto-repair.
- **Failure Rate** — Percentage still failing after repair.
- **Latency (ms)** — Average generation time.
- **Token Efficiency** — Passes per token generated.

Metrics are available **per category/domain** with **model-wise comparisons** in the analytics dashboard.

---

## Research Impact

FailProof-LLM enables a systematic, reproducible evaluation of LLMs under adversarial and domain-specific constraints.  
By unifying **data generation, evaluation, repair, and analytics**, the platform supports:

- **Benchmark creation** for model comparison
- **Robustness testing** for deployment readiness
- **Model improvement** via targeted feedback


