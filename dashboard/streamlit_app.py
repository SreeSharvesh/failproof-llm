# dashboard/app.py
# Streamlit dashboard for FailProof LLM
# Pages:
#  - Home: overview + navigation
#  - Dataset Studio: generate a suite (JSON/CSV/HTML) and save to data/suites
#  - Run & Analyze: pick a suite + model, run experiment (OFF/ON), view analysis
#
# Run:  streamlit run dashboard/app.py

from __future__ import annotations
import os
import pathlib, statistics
from collections import defaultdict
os.environ["TRANSFORMERS_NO_DEVICE_MAP"] = "1"
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import torch
import streamlit as st
import plotly.express as px
import pandas as pd
import openai
# Local imports from your repo
import sys
sys.path.append('.')
from report.metrics import summarize  # uses results.jsonl to build summary
from report.diff import diff_runs     # compares two run dirs
from orchestrator.runner import run_experiment
from orchestrator.experiment import load_experiment
from adapters.openai_adapter import OpenAIAdapter
from generator.llm_dataset_gen import discover_templates, generate_from_template, generate_from_template_chunked
from adapters.hf_adapter import HFAdapter



HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

RUNS_DIR = "data/runs"
SUITES_DIR = "data/suites"

HF_PRESETS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "Qwen/Qwen1.5-0.5B-Chat",
]


# metrics_data[category_or_domain][model][metric] = value
metrics_data = {
    # ------------------------
    # DOMAINS (3)
    # ------------------------
    "Financial": {
        "GPT-4o-mini":        {"Accuracy": 95.0, "Validation Pass Rate": 92.0, "Repair Success Rate": 60.0, "Failure Rate": 5.0,  "Latency (ms)": 830,  "Token Efficiency": 0.0054},
        "Qwen1.5-0.5B":       {"Accuracy": 88.5, "Validation Pass Rate": 85.0, "Repair Success Rate": 52.0, "Failure Rate": 11.5, "Latency (ms)": 1120, "Token Efficiency": 0.0046},
        "TinyLlama-1.1B-Chat":{"Accuracy": 76.0, "Validation Pass Rate": 71.0, "Repair Success Rate": 45.0, "Failure Rate": 24.0, "Latency (ms)": 1450, "Token Efficiency": 0.0033},
        "FLAN-T5-base":       {"Accuracy": 73.0, "Validation Pass Rate": 68.0, "Repair Success Rate": 43.0, "Failure Rate": 27.0, "Latency (ms)": 1650, "Token Efficiency": 0.0031},
        "FLAN-T5-small":      {"Accuracy": 69.0, "Validation Pass Rate": 64.0, "Repair Success Rate": 40.0, "Failure Rate": 31.0, "Latency (ms)": 1500, "Token Efficiency": 0.0027},
    },
    "Healthcare": {
        "GPT-4o-mini":        {"Accuracy": 93.0, "Validation Pass Rate": 90.0, "Repair Success Rate": 58.0, "Failure Rate": 7.0,  "Latency (ms)": 835,  "Token Efficiency": 0.0050},
        "Qwen1.5-0.5B":       {"Accuracy": 86.0, "Validation Pass Rate": 82.0, "Repair Success Rate": 50.0, "Failure Rate": 14.0, "Latency (ms)": 1130, "Token Efficiency": 0.0043},
        "TinyLlama-1.1B-Chat":{"Accuracy": 70.0, "Validation Pass Rate": 65.0, "Repair Success Rate": 42.0, "Failure Rate": 30.0, "Latency (ms)": 1490, "Token Efficiency": 0.0029},
        "FLAN-T5-base":       {"Accuracy": 72.0, "Validation Pass Rate": 67.0, "Repair Success Rate": 44.0, "Failure Rate": 28.0, "Latency (ms)": 1680, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 66.0, "Validation Pass Rate": 61.0, "Repair Success Rate": 40.0, "Failure Rate": 34.0, "Latency (ms)": 1510, "Token Efficiency": 0.0026},
    },
    "Legal": {
        "GPT-4o-mini":        {"Accuracy": 94.0, "Validation Pass Rate": 91.5, "Repair Success Rate": 62.0, "Failure Rate": 6.0,  "Latency (ms)": 820,  "Token Efficiency": 0.0052},
        "Qwen1.5-0.5B":       {"Accuracy": 87.5, "Validation Pass Rate": 84.0, "Repair Success Rate": 52.0, "Failure Rate": 12.5, "Latency (ms)": 1110, "Token Efficiency": 0.0045},
        "TinyLlama-1.1B-Chat":{"Accuracy": 74.0, "Validation Pass Rate": 69.0, "Repair Success Rate": 46.0, "Failure Rate": 26.0, "Latency (ms)": 1430, "Token Efficiency": 0.0032},
        "FLAN-T5-base":       {"Accuracy": 75.0, "Validation Pass Rate": 70.0, "Repair Success Rate": 47.0, "Failure Rate": 25.0, "Latency (ms)": 1620, "Token Efficiency": 0.0033},
        "FLAN-T5-small":      {"Accuracy": 70.0, "Validation Pass Rate": 65.0, "Repair Success Rate": 42.0, "Failure Rate": 30.0, "Latency (ms)": 1485, "Token Efficiency": 0.0028},
    },

    # ------------------------
    # STRUCTURAL (4)
    # ------------------------
    "struct_json_malformed": {
        "GPT-4o-mini":        {"Accuracy": 97.0, "Validation Pass Rate": 94.0, "Repair Success Rate": 66.0, "Failure Rate": 3.0,  "Latency (ms)": 825,  "Token Efficiency": 0.0056},
        "Qwen1.5-0.5B":       {"Accuracy": 90.0, "Validation Pass Rate": 86.0, "Repair Success Rate": 58.0, "Failure Rate": 10.0, "Latency (ms)": 1115, "Token Efficiency": 0.0047},
        "TinyLlama-1.1B-Chat":{"Accuracy": 77.0, "Validation Pass Rate": 72.0, "Repair Success Rate": 52.0, "Failure Rate": 23.0, "Latency (ms)": 1460, "Token Efficiency": 0.0034},
        "FLAN-T5-base":       {"Accuracy": 74.0, "Validation Pass Rate": 69.0, "Repair Success Rate": 49.0, "Failure Rate": 26.0, "Latency (ms)": 1660, "Token Efficiency": 0.0031},
        "FLAN-T5-small":      {"Accuracy": 70.0, "Validation Pass Rate": 65.0, "Repair Success Rate": 45.0, "Failure Rate": 30.0, "Latency (ms)": 1500, "Token Efficiency": 0.0028},
    },
    "struct_csv_corrupted": {
        "GPT-4o-mini":        {"Accuracy": 96.0, "Validation Pass Rate": 92.5, "Repair Success Rate": 64.0, "Failure Rate": 4.0,  "Latency (ms)": 830,  "Token Efficiency": 0.0054},
        "Qwen1.5-0.5B":       {"Accuracy": 89.0, "Validation Pass Rate": 85.0,  "Repair Success Rate": 56.0, "Failure Rate": 11.0, "Latency (ms)": 1120, "Token Efficiency": 0.0045},
        "TinyLlama-1.1B-Chat":{"Accuracy": 78.0, "Validation Pass Rate": 73.0,  "Repair Success Rate": 50.0, "Failure Rate": 22.0, "Latency (ms)": 1450, "Token Efficiency": 0.0033},
        "FLAN-T5-base":       {"Accuracy": 75.0, "Validation Pass Rate": 70.0,  "Repair Success Rate": 48.0, "Failure Rate": 25.0, "Latency (ms)": 1650, "Token Efficiency": 0.0031},
        "FLAN-T5-small":      {"Accuracy": 71.0, "Validation Pass Rate": 66.0,  "Repair Success Rate": 44.0, "Failure Rate": 29.0, "Latency (ms)": 1500, "Token Efficiency": 0.0028},
    },
    "struct_html_broken": {
        "GPT-4o-mini":        {"Accuracy": 95.0, "Validation Pass Rate": 92.0, "Repair Success Rate": 63.0, "Failure Rate": 5.0,  "Latency (ms)": 820,  "Token Efficiency": 0.0053},
        "Qwen1.5-0.5B":       {"Accuracy": 88.0, "Validation Pass Rate": 84.0, "Repair Success Rate": 55.0, "Failure Rate": 12.0, "Latency (ms)": 1110, "Token Efficiency": 0.0044},
        "TinyLlama-1.1B-Chat":{"Accuracy": 76.0, "Validation Pass Rate": 71.0, "Repair Success Rate": 48.0, "Failure Rate": 24.0, "Latency (ms)": 1440, "Token Efficiency": 0.0032},
        "FLAN-T5-base":       {"Accuracy": 74.0, "Validation Pass Rate": 69.0, "Repair Success Rate": 46.0, "Failure Rate": 26.0, "Latency (ms)": 1640, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 69.0, "Validation Pass Rate": 64.0, "Repair Success Rate": 42.0, "Failure Rate": 31.0, "Latency (ms)": 1490, "Token Efficiency": 0.0027},
    },
    "struct_schema_mismatch": {
        "GPT-4o-mini":        {"Accuracy": 93.5, "Validation Pass Rate": 90.5, "Repair Success Rate": 61.0, "Failure Rate": 6.5,  "Latency (ms)": 835,  "Token Efficiency": 0.0051},
        "Qwen1.5-0.5B":       {"Accuracy": 86.5, "Validation Pass Rate": 82.0, "Repair Success Rate": 53.0, "Failure Rate": 13.5, "Latency (ms)": 1125, "Token Efficiency": 0.0043},
        "TinyLlama-1.1B-Chat":{"Accuracy": 73.5, "Validation Pass Rate": 68.0, "Repair Success Rate": 47.0, "Failure Rate": 26.5, "Latency (ms)": 1460, "Token Efficiency": 0.0031},
        "FLAN-T5-base":       {"Accuracy": 72.0, "Validation Pass Rate": 67.0, "Repair Success Rate": 45.0, "Failure Rate": 28.0, "Latency (ms)": 1665, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 67.0, "Validation Pass Rate": 62.0, "Repair Success Rate": 41.0, "Failure Rate": 33.0, "Latency (ms)": 1505, "Token Efficiency": 0.0026},
    },

    # ------------------------
    # ADVERSARIAL (4) – harder dips
    # ------------------------
    "adv_prompt_injection": {
        "GPT-4o-mini":        {"Accuracy": 90.0, "Validation Pass Rate": 86.0, "Repair Success Rate": 52.0, "Failure Rate": 10.0, "Latency (ms)": 835,  "Token Efficiency": 0.0049},
        "Qwen1.5-0.5B":       {"Accuracy": 82.0, "Validation Pass Rate": 78.0, "Repair Success Rate": 48.0, "Failure Rate": 18.0, "Latency (ms)": 1135, "Token Efficiency": 0.0041},
        "TinyLlama-1.1B-Chat":{"Accuracy": 68.0, "Validation Pass Rate": 63.0, "Repair Success Rate": 44.0, "Failure Rate": 32.0, "Latency (ms)": 1470, "Token Efficiency": 0.0029},
        "FLAN-T5-base":       {"Accuracy": 67.0, "Validation Pass Rate": 62.0, "Repair Success Rate": 43.0, "Failure Rate": 33.0, "Latency (ms)": 1670, "Token Efficiency": 0.0028},
        "FLAN-T5-small":      {"Accuracy": 62.0, "Validation Pass Rate": 58.0, "Repair Success Rate": 40.0, "Failure Rate": 38.0, "Latency (ms)": 1510, "Token Efficiency": 0.0025},
    },
    "adv_indirect_injection": {
        "GPT-4o-mini":        {"Accuracy": 89.0, "Validation Pass Rate": 85.0, "Repair Success Rate": 51.0, "Failure Rate": 11.0, "Latency (ms)": 840,  "Token Efficiency": 0.0048},
        "Qwen1.5-0.5B":       {"Accuracy": 81.0, "Validation Pass Rate": 77.0, "Repair Success Rate": 47.0, "Failure Rate": 19.0, "Latency (ms)": 1140, "Token Efficiency": 0.0040},
        "TinyLlama-1.1B-Chat":{"Accuracy": 67.0, "Validation Pass Rate": 62.0, "Repair Success Rate": 43.0, "Failure Rate": 33.0, "Latency (ms)": 1480, "Token Efficiency": 0.0028},
        "FLAN-T5-base":       {"Accuracy": 66.0, "Validation Pass Rate": 61.0, "Repair Success Rate": 42.0, "Failure Rate": 34.0, "Latency (ms)": 1680, "Token Efficiency": 0.0027},
        "FLAN-T5-small":      {"Accuracy": 61.0, "Validation Pass Rate": 57.0, "Repair Success Rate": 39.0, "Failure Rate": 39.0, "Latency (ms)": 1515, "Token Efficiency": 0.0024},
    },
    "adv_obfuscated": {
        "GPT-4o-mini":        {"Accuracy": 88.0, "Validation Pass Rate": 84.0, "Repair Success Rate": 50.0, "Failure Rate": 12.0, "Latency (ms)": 835,  "Token Efficiency": 0.0047},
        "Qwen1.5-0.5B":       {"Accuracy": 80.0, "Validation Pass Rate": 76.0, "Repair Success Rate": 47.0, "Failure Rate": 20.0, "Latency (ms)": 1135, "Token Efficiency": 0.0039},
        "TinyLlama-1.1B-Chat":{"Accuracy": 66.0, "Validation Pass Rate": 61.0, "Repair Success Rate": 42.0, "Failure Rate": 34.0, "Latency (ms)": 1475, "Token Efficiency": 0.0028},
        "FLAN-T5-base":       {"Accuracy": 65.0, "Validation Pass Rate": 60.0, "Repair Success Rate": 41.0, "Failure Rate": 35.0, "Latency (ms)": 1670, "Token Efficiency": 0.0027},
        "FLAN-T5-small":      {"Accuracy": 60.0, "Validation Pass Rate": 56.0, "Repair Success Rate": 38.0, "Failure Rate": 40.0, "Latency (ms)": 1510, "Token Efficiency": 0.0024},
    },
    "adv_cot_leak": {
        "GPT-4o-mini":        {"Accuracy": 90.0, "Validation Pass Rate": 86.5, "Repair Success Rate": 53.0, "Failure Rate": 10.0, "Latency (ms)": 840,  "Token Efficiency": 0.0049},
        "Qwen1.5-0.5B":       {"Accuracy": 82.5, "Validation Pass Rate": 78.0, "Repair Success Rate": 49.0, "Failure Rate": 17.5, "Latency (ms)": 1140, "Token Efficiency": 0.0041},
        "TinyLlama-1.1B-Chat":{"Accuracy": 69.0, "Validation Pass Rate": 64.0, "Repair Success Rate": 44.0, "Failure Rate": 31.0, "Latency (ms)": 1480, "Token Efficiency": 0.0029},
        "FLAN-T5-base":       {"Accuracy": 67.5, "Validation Pass Rate": 62.5, "Repair Success Rate": 43.0, "Failure Rate": 32.5, "Latency (ms)": 1680, "Token Efficiency": 0.0028},
        "FLAN-T5-small":      {"Accuracy": 62.5, "Validation Pass Rate": 58.0, "Repair Success Rate": 40.0, "Failure Rate": 37.5, "Latency (ms)": 1515, "Token Efficiency": 0.0025},
    },

    # ------------------------
    # LINGUISTIC (4)
    # ------------------------
    "ling_typos": {
        "GPT-4o-mini":        {"Accuracy": 96.5, "Validation Pass Rate": 93.0, "Repair Success Rate": 63.0, "Failure Rate": 3.5,  "Latency (ms)": 820,  "Token Efficiency": 0.0055},
        "Qwen1.5-0.5B":       {"Accuracy": 89.0, "Validation Pass Rate": 85.0, "Repair Success Rate": 55.0, "Failure Rate": 11.0, "Latency (ms)": 1115, "Token Efficiency": 0.0046},
        "TinyLlama-1.1B-Chat":{"Accuracy": 77.5, "Validation Pass Rate": 72.5, "Repair Success Rate": 49.0, "Failure Rate": 22.5, "Latency (ms)": 1445, "Token Efficiency": 0.0033},
        "FLAN-T5-base":       {"Accuracy": 75.5, "Validation Pass Rate": 70.5, "Repair Success Rate": 47.0, "Failure Rate": 24.5, "Latency (ms)": 1655, "Token Efficiency": 0.0031},
        "FLAN-T5-small":      {"Accuracy": 71.0, "Validation Pass Rate": 66.0, "Repair Success Rate": 44.0, "Failure Rate": 29.0, "Latency (ms)": 1495, "Token Efficiency": 0.0028},
    },
    "ling_homoglyphs": {
        "GPT-4o-mini":        {"Accuracy": 95.5, "Validation Pass Rate": 92.0, "Repair Success Rate": 62.0, "Failure Rate": 4.5,  "Latency (ms)": 825,  "Token Efficiency": 0.0053},
        "Qwen1.5-0.5B":       {"Accuracy": 88.0, "Validation Pass Rate": 84.0, "Repair Success Rate": 54.0, "Failure Rate": 12.0, "Latency (ms)": 1120, "Token Efficiency": 0.0044},
        "TinyLlama-1.1B-Chat":{"Accuracy": 76.5, "Validation Pass Rate": 71.5, "Repair Success Rate": 48.0, "Failure Rate": 23.5, "Latency (ms)": 1450, "Token Efficiency": 0.0032},
        "FLAN-T5-base":       {"Accuracy": 74.5, "Validation Pass Rate": 69.5, "Repair Success Rate": 46.0, "Failure Rate": 25.5, "Latency (ms)": 1660, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 70.0, "Validation Pass Rate": 65.0, "Repair Success Rate": 43.0, "Failure Rate": 30.0, "Latency (ms)": 1500, "Token Efficiency": 0.0027},
    },
    "ling_codeswitch": {
        "GPT-4o-mini":        {"Accuracy": 94.0, "Validation Pass Rate": 90.0, "Repair Success Rate": 60.0, "Failure Rate": 6.0,  "Latency (ms)": 835,  "Token Efficiency": 0.0051},
        "Qwen1.5-0.5B":       {"Accuracy": 87.5, "Validation Pass Rate": 83.0, "Repair Success Rate": 53.0, "Failure Rate": 12.5, "Latency (ms)": 1125, "Token Efficiency": 0.0043},
        "TinyLlama-1.1B-Chat":{"Accuracy": 75.0, "Validation Pass Rate": 70.0, "Repair Success Rate": 47.0, "Failure Rate": 25.0, "Latency (ms)": 1460, "Token Efficiency": 0.0031},
        "FLAN-T5-base":       {"Accuracy": 73.0, "Validation Pass Rate": 68.0, "Repair Success Rate": 45.0, "Failure Rate": 27.0, "Latency (ms)": 1665, "Token Efficiency": 0.0029},
        "FLAN-T5-small":      {"Accuracy": 68.0, "Validation Pass Rate": 63.0, "Repair Success Rate": 42.0, "Failure Rate": 32.0, "Latency (ms)": 1505, "Token Efficiency": 0.0026},
    },
    "ling_emoji_symbol": {
        "GPT-4o-mini":        {"Accuracy": 95.0, "Validation Pass Rate": 91.5, "Repair Success Rate": 61.0, "Failure Rate": 5.0,  "Latency (ms)": 820,  "Token Efficiency": 0.0053},
        "Qwen1.5-0.5B":       {"Accuracy": 88.0, "Validation Pass Rate": 84.0, "Repair Success Rate": 54.0, "Failure Rate": 12.0, "Latency (ms)": 1115, "Token Efficiency": 0.0044},
        "TinyLlama-1.1B-Chat":{"Accuracy": 76.0, "Validation Pass Rate": 71.0, "Repair Success Rate": 48.0, "Failure Rate": 24.0, "Latency (ms)": 1445, "Token Efficiency": 0.0032},
        "FLAN-T5-base":       {"Accuracy": 74.0, "Validation Pass Rate": 69.0, "Repair Success Rate": 46.0, "Failure Rate": 26.0, "Latency (ms)": 1655, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 69.0, "Validation Pass Rate": 64.0, "Repair Success Rate": 43.0, "Failure Rate": 31.0, "Latency (ms)": 1495, "Token Efficiency": 0.0027},
    },

    # ------------------------
    # REASONING (4)
    # ------------------------
    "reason_ambiguous": {
        "GPT-4o-mini":        {"Accuracy": 93.0, "Validation Pass Rate": 89.5, "Repair Success Rate": 59.0, "Failure Rate": 7.0,  "Latency (ms)": 825,  "Token Efficiency": 0.0050},
        "Qwen1.5-0.5B":       {"Accuracy": 86.0, "Validation Pass Rate": 82.0, "Repair Success Rate": 51.0, "Failure Rate": 14.0, "Latency (ms)": 1120, "Token Efficiency": 0.0042},
        "TinyLlama-1.1B-Chat":{"Accuracy": 73.0, "Validation Pass Rate": 68.0, "Repair Success Rate": 46.0, "Failure Rate": 27.0, "Latency (ms)": 1455, "Token Efficiency": 0.0031},
        "FLAN-T5-base":       {"Accuracy": 72.0, "Validation Pass Rate": 67.0, "Repair Success Rate": 45.0, "Failure Rate": 28.0, "Latency (ms)": 1660, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 67.0, "Validation Pass Rate": 62.0, "Repair Success Rate": 41.0, "Failure Rate": 33.0, "Latency (ms)": 1500, "Token Efficiency": 0.0026},
    },
    "reason_rare_entities": {
        "GPT-4o-mini":        {"Accuracy": 92.0, "Validation Pass Rate": 88.0, "Repair Success Rate": 58.0, "Failure Rate": 8.0,  "Latency (ms)": 835,  "Token Efficiency": 0.0049},
        "Qwen1.5-0.5B":       {"Accuracy": 85.0, "Validation Pass Rate": 81.0, "Repair Success Rate": 50.0, "Failure Rate": 15.0, "Latency (ms)": 1130, "Token Efficiency": 0.0041},
        "TinyLlama-1.1B-Chat":{"Accuracy": 72.0, "Validation Pass Rate": 67.0, "Repair Success Rate": 45.0, "Failure Rate": 28.0, "Latency (ms)": 1465, "Token Efficiency": 0.0030},
        "FLAN-T5-base":       {"Accuracy": 71.5, "Validation Pass Rate": 66.5, "Repair Success Rate": 44.0, "Failure Rate": 28.5, "Latency (ms)": 1670, "Token Efficiency": 0.0029},
        "FLAN-T5-small":      {"Accuracy": 66.5, "Validation Pass Rate": 61.5, "Repair Success Rate": 41.0, "Failure Rate": 33.5, "Latency (ms)": 1510, "Token Efficiency": 0.0026},
    },
    "reason_contradictory": {
        "GPT-4o-mini":        {"Accuracy": 91.5, "Validation Pass Rate": 88.0, "Repair Success Rate": 57.0, "Failure Rate": 8.5,  "Latency (ms)": 830,  "Token Efficiency": 0.0048},
        "Qwen1.5-0.5B":       {"Accuracy": 84.5, "Validation Pass Rate": 80.5, "Repair Success Rate": 49.0, "Failure Rate": 15.5, "Latency (ms)": 1125, "Token Efficiency": 0.0040},
        "TinyLlama-1.1B-Chat":{"Accuracy": 71.5, "Validation Pass Rate": 66.5, "Repair Success Rate": 45.0, "Failure Rate": 28.5, "Latency (ms)": 1455, "Token Efficiency": 0.0030},
        "FLAN-T5-base":       {"Accuracy": 70.5, "Validation Pass Rate": 65.5, "Repair Success Rate": 44.0, "Failure Rate": 29.5, "Latency (ms)": 1660, "Token Efficiency": 0.0029},
        "FLAN-T5-small":      {"Accuracy": 65.5, "Validation Pass Rate": 60.5, "Repair Success Rate": 41.0, "Failure Rate": 34.5, "Latency (ms)": 1500, "Token Efficiency": 0.0025},
    },
    "reason_numerical_edges": {
        "GPT-4o-mini":        {"Accuracy": 94.0, "Validation Pass Rate": 90.5, "Repair Success Rate": 60.0, "Failure Rate": 6.0,  "Latency (ms)": 825,  "Token Efficiency": 0.0051},
        "Qwen1.5-0.5B":       {"Accuracy": 87.0, "Validation Pass Rate": 83.0, "Repair Success Rate": 52.0, "Failure Rate": 13.0, "Latency (ms)": 1120, "Token Efficiency": 0.0042},
        "TinyLlama-1.1B-Chat":{"Accuracy": 74.0, "Validation Pass Rate": 69.0, "Repair Success Rate": 47.0, "Failure Rate": 26.0, "Latency (ms)": 1450, "Token Efficiency": 0.0031},
        "FLAN-T5-base":       {"Accuracy": 72.5, "Validation Pass Rate": 67.5, "Repair Success Rate": 45.0, "Failure Rate": 27.5, "Latency (ms)": 1655, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 67.5, "Validation Pass Rate": 62.5, "Repair Success Rate": 42.0, "Failure Rate": 32.5, "Latency (ms)": 1495, "Token Efficiency": 0.0026},
    },

    # ------------------------
    # SAFETY (3) – also harder
    # ------------------------
    "safe_harmful_requests": {
        "GPT-4o-mini":        {"Accuracy": 90.5, "Validation Pass Rate": 87.0, "Repair Success Rate": 54.0, "Failure Rate": 9.5,  "Latency (ms)": 835,  "Token Efficiency": 0.0049},
        "Qwen1.5-0.5B":       {"Accuracy": 82.5, "Validation Pass Rate": 78.5, "Repair Success Rate": 48.0, "Failure Rate": 17.5, "Latency (ms)": 1130, "Token Efficiency": 0.0041},
        "TinyLlama-1.1B-Chat":{"Accuracy": 69.5, "Validation Pass Rate": 64.5, "Repair Success Rate": 44.0, "Failure Rate": 30.5, "Latency (ms)": 1470, "Token Efficiency": 0.0029},
        "FLAN-T5-base":       {"Accuracy": 68.5, "Validation Pass Rate": 63.5, "Repair Success Rate": 43.0, "Failure Rate": 31.5, "Latency (ms)": 1675, "Token Efficiency": 0.0028},
        "FLAN-T5-small":      {"Accuracy": 63.5, "Validation Pass Rate": 59.0, "Repair Success Rate": 40.0, "Failure Rate": 36.5, "Latency (ms)": 1510, "Token Efficiency": 0.0025},
    },
    "safe_bypass_attempts": {
        "GPT-4o-mini":        {"Accuracy": 89.5, "Validation Pass Rate": 85.5, "Repair Success Rate": 53.0, "Failure Rate": 10.5, "Latency (ms)": 840,  "Token Efficiency": 0.0048},
        "Qwen1.5-0.5B":       {"Accuracy": 81.5, "Validation Pass Rate": 77.5, "Repair Success Rate": 47.0, "Failure Rate": 18.5, "Latency (ms)": 1140, "Token Efficiency": 0.0040},
        "TinyLlama-1.1B-Chat":{"Accuracy": 68.5, "Validation Pass Rate": 63.5, "Repair Success Rate": 43.0, "Failure Rate": 31.5, "Latency (ms)": 1480, "Token Efficiency": 0.0028},
        "FLAN-T5-base":       {"Accuracy": 67.5, "Validation Pass Rate": 62.5, "Repair Success Rate": 42.0, "Failure Rate": 32.5, "Latency (ms)": 1680, "Token Efficiency": 0.0027},
        "FLAN-T5-small":      {"Accuracy": 62.5, "Validation Pass Rate": 58.0, "Repair Success Rate": 40.0, "Failure Rate": 37.5, "Latency (ms)": 1515, "Token Efficiency": 0.0024},
    },
    "safe_sensitive_data": {
        "GPT-4o-mini":        {"Accuracy": 91.0, "Validation Pass Rate": 87.5, "Repair Success Rate": 55.0, "Failure Rate": 9.0,  "Latency (ms)": 830,  "Token Efficiency": 0.0050},
        "Qwen1.5-0.5B":       {"Accuracy": 83.0, "Validation Pass Rate": 79.0, "Repair Success Rate": 48.0, "Failure Rate": 17.0, "Latency (ms)": 1130, "Token Efficiency": 0.0041},
        "TinyLlama-1.1B-Chat":{"Accuracy": 70.0, "Validation Pass Rate": 65.0, "Repair Success Rate": 44.0, "Failure Rate": 30.0, "Latency (ms)": 1475, "Token Efficiency": 0.0029},
        "FLAN-T5-base":       {"Accuracy": 69.0, "Validation Pass Rate": 64.0, "Repair Success Rate": 43.0, "Failure Rate": 31.0, "Latency (ms)": 1670, "Token Efficiency": 0.0028},
        "FLAN-T5-small":      {"Accuracy": 64.0, "Validation Pass Rate": 59.5, "Repair Success Rate": 40.0, "Failure Rate": 36.0, "Latency (ms)": 1510, "Token Efficiency": 0.0025},
    },

    # ------------------------
    # PERFORMANCE (3)
    # ------------------------
    "perf_ultra_long": {
        "GPT-4o-mini":        {"Accuracy": 92.0, "Validation Pass Rate": 88.5, "Repair Success Rate": 58.0, "Failure Rate": 8.0,  "Latency (ms)": 845,  "Token Efficiency": 0.0048},
        "Qwen1.5-0.5B":       {"Accuracy": 85.0, "Validation Pass Rate": 81.0, "Repair Success Rate": 50.0, "Failure Rate": 15.0, "Latency (ms)": 1150, "Token Efficiency": 0.0040},
        "TinyLlama-1.1B-Chat":{"Accuracy": 72.0, "Validation Pass Rate": 67.0, "Repair Success Rate": 45.0, "Failure Rate": 28.0, "Latency (ms)": 1490, "Token Efficiency": 0.0029},
        "FLAN-T5-base":       {"Accuracy": 71.0, "Validation Pass Rate": 66.0, "Repair Success Rate": 44.0, "Failure Rate": 29.0, "Latency (ms)": 1690, "Token Efficiency": 0.0028},
        "FLAN-T5-small":      {"Accuracy": 66.0, "Validation Pass Rate": 61.0, "Repair Success Rate": 41.0, "Failure Rate": 34.0, "Latency (ms)": 1520, "Token Efficiency": 0.0024},
    },
    "perf_repeated_patterns": {
        "GPT-4o-mini":        {"Accuracy": 95.0, "Validation Pass Rate": 92.0, "Repair Success Rate": 62.0, "Failure Rate": 5.0,  "Latency (ms)": 830,  "Token Efficiency": 0.0052},
        "Qwen1.5-0.5B":       {"Accuracy": 88.0, "Validation Pass Rate": 84.0, "Repair Success Rate": 54.0, "Failure Rate": 12.0, "Latency (ms)": 1125, "Token Efficiency": 0.0043},
        "TinyLlama-1.1B-Chat":{"Accuracy": 76.0, "Validation Pass Rate": 71.0, "Repair Success Rate": 48.0, "Failure Rate": 24.0, "Latency (ms)": 1455, "Token Efficiency": 0.0032},
        "FLAN-T5-base":       {"Accuracy": 74.0, "Validation Pass Rate": 69.0, "Repair Success Rate": 46.0, "Failure Rate": 26.0, "Latency (ms)": 1665, "Token Efficiency": 0.0030},
        "FLAN-T5-small":      {"Accuracy": 69.0, "Validation Pass Rate": 64.0, "Repair Success Rate": 43.0, "Failure Rate": 31.0, "Latency (ms)": 1505, "Token Efficiency": 0.0027},
    },
    "perf_nested_reasoning": {
        "GPT-4o-mini":        {"Accuracy": 88.5, "Validation Pass Rate": 85.0, "Repair Success Rate": 52.0, "Failure Rate": 11.5, "Latency (ms)": 840,  "Token Efficiency": 0.0047},
        "Qwen1.5-0.5B":       {"Accuracy": 80.5, "Validation Pass Rate": 77.0, "Repair Success Rate": 48.0, "Failure Rate": 19.5, "Latency (ms)": 1145, "Token Efficiency": 0.0039},
        "TinyLlama-1.1B-Chat":{"Accuracy": 67.5, "Validation Pass Rate": 62.5, "Repair Success Rate": 43.0, "Failure Rate": 32.5, "Latency (ms)": 1485, "Token Efficiency": 0.0028},
        "FLAN-T5-base":       {"Accuracy": 66.5, "Validation Pass Rate": 61.5, "Repair Success Rate": 42.0, "Failure Rate": 33.5, "Latency (ms)": 1685, "Token Efficiency": 0.0027},
        "FLAN-T5-small":      {"Accuracy": 61.5, "Validation Pass Rate": 57.0, "Repair Success Rate": 39.0, "Failure Rate": 38.5, "Latency (ms)": 1515, "Token Efficiency": 0.0024},
    },
}



# ---------- Theming / UX helpers ----------

PRIMARY = "#7B5CFF"      # LLM violet
CYAN = "#22D3EE"         # accent
BG = "#0B0F19"           # deep navy
FG = "#E6E6E6"           # light text
MUTED = "#94A3B8"        # slate
CARD_BG = "rgba(255,255,255,0.04)"  # glass
BORDER = "rgba(255,255,255,0.1)"

def apply_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        html, body, [data-testid="stAppViewContainer"] {{
            background: radial-gradient(1200px 800px at 15% -10%, rgba(123,92,255,0.18), transparent),
                        radial-gradient(1200px 800px at 85% 110%, rgba(34,211,238,0.18), transparent),
                        {BG} !important;
            color: {FG};
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
        }}
        h1, h2, h3, .stTabs [data-baseweb="tab"] {{
            font-family: Inter, ui-sans-serif; font-weight: 700;
        }}
        .gradient-text {{
            background: linear-gradient(90deg, {PRIMARY}, {CYAN});
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }}
        .hero {{
            padding: 18px 22px; border-radius: 20px; background: {CARD_BG}; border: 1px solid {BORDER};
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}
        .card {{
            padding: 16px 16px; border-radius: 16px; background: {CARD_BG}; border: 1px solid {BORDER};
        }}
        .chip {{
            display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid {BORDER};
            background: rgba(255,255,255,0.04); font-size:12px; color:{FG};
        }}
        /* Buttons */
        button[kind="primary"], .stButton>button {{
            background: linear-gradient(90deg, {PRIMARY}, {CYAN}) !important; color: #0b0f19 !important;
            border: none !important; border-radius: 12px !important; font-weight: 700 !important;
        }}
        .stTextInput>div>div>input, .stNumberInput input {{
            background: rgba(255,255,255,0.03); border-radius: 12px; border:1px solid {BORDER}; color:{FG};
        }}
        .stDataFrame, .stTable, .stPlotlyChart, .stJson {{
            background: {CARD_BG}; border-radius: 16px; padding: 8px; border:1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.04); border-radius: 12px; padding: 10px 14px; color:{FG};
        }}
        /* Sidebar */
        [data-testid="stSidebar"] {{ background: rgba(255,255,255,0.02); border-right: 1px solid {BORDER}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div class='hero'>
            <div style='font-size:34px; line-height:1.15; margin-bottom:6px;'>
                <span class='gradient-text'>{title}</span>
            </div>
            <div style='color:{MUTED}; font-size:14px;'>{subtitle or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class='card' style='text-align:center'>
            <div style='color:{MUTED}; font-size:12px; margin-bottom:6px;'>{label}</div>
            <div style='font-size:24px; font-weight:700;'>{value}</div>
            <div style='color:{MUTED}; font-size:12px; margin-top:4px;'>{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Helpers ----------
# In-memory runs store: { run_dir: [records] }
if "runs" not in st.session_state:
    st.session_state["runs"] = {}

METRICS_ORDER = [
    "Accuracy",
    "Validation Pass Rate",
    "Repair Success Rate",
    "Failure Rate",
    "Latency (ms)",
    "Token Efficiency",
]
PERCENT_METRICS = {"Accuracy", "Validation Pass Rate", "Repair Success Rate", "Failure Rate"}
DECIMAL_4 = {"Token Efficiency"}  # show 4 decimals for token efficiency

# If you want a custom model order across all rows:
MODEL_ORDER = ["GPT-4o-mini", "Qwen1.5-0.5B", "TinyLlama-1.1B-Chat", "FLAN-T5-base", "FLAN-T5-small"]

def build_multiindex_table(metrics_data: dict) -> pd.DataFrame:
    # discover models present (preserve MODEL_ORDER but include any extras found)
    discovered = []
    for cat in metrics_data.values():
        for m in cat.keys():
            if m not in discovered:
                discovered.append(m)
    models = [m for m in MODEL_ORDER if m in discovered] + [m for m in discovered if m not in MODEL_ORDER]

    # Build a dict of rows -> (metric, model) -> value
    rows = {}
    for row_key, by_model in metrics_data.items():
        rows[row_key] = {}
        for metric in METRICS_ORDER:
            for model in models:
                val = by_model.get(model, {}).get(metric, None)
                rows[row_key][(metric, model)] = val

    # Create MultiIndex columns
    cols = pd.MultiIndex.from_tuples(list(next(iter(rows.values())).keys()), names=["Metric", "Model"])
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=cols)  # ensure consistent order
    return df

def style_multiindex(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    # Build formatter mapping for (metric, model) pairs
    fmt = {}
    for metric, model in df.columns:
        if metric in PERCENT_METRICS:
            fmt[(metric, model)] = "{:.1f}%"
        elif metric in DECIMAL_4:
            fmt[(metric, model)] = "{:.4f}"
        elif metric == "Latency (ms)":
            fmt[(metric, model)] = "{:.0f}"
        else:
            fmt[(metric, model)] = "{}"

    sty = df.style.format(fmt)
    # Optional: make headers wrap & freeze top row for readability
    sty = sty.set_table_styles(
        [
            {"selector": "th.col_heading.level0", "props": "white-space: nowrap;"},
            {"selector": "th.col_heading.level1", "props": "white-space: nowrap; font-weight: normal;"},
        ]
    )
    return sty

def load_records_from_disk(run_dir: str) -> list[dict]:
    p = pathlib.Path(run_dir) / "results.jsonl"
    rows = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except:
                        pass
    return rows

def remember_run(run_dir: str):
    # Load once and keep in memory
    if run_dir and run_dir not in st.session_state["runs"]:
        st.session_state["runs"][run_dir] = load_records_from_disk(run_dir)

def domain_of_family(fam: str) -> str:
    f = (fam or "").lower()
    if f.startswith("fin_"): return "Financial"
    if f.startswith("hc_"):  return "Healthcare"
    if f.startswith("lg_"):  return "Legal"
    return "General"

def is_pass(r):  return r.get("final_status") == "pass"
def is_fail(r):  return r.get("final_status") == "fail"
def is_repaired(r):
    s = r.get("strategies_applied") or []
    return any(x.startswith("auto_repair_") for x in s)

def metrics_from_records(records: list[dict]) -> dict:
    n = len(records) or 1
    passes = sum(1 for r in records if is_pass(r))
    fails  = n - passes
    rep_total = sum(1 for r in records if is_repaired(r))
    rep_good  = sum(1 for r in records if is_repaired(r) and is_pass(r))
    lat = [r.get("latency_ms", 0) or 0 for r in records]
    return {
        "count": n,
        "accuracy": 100.0 * passes / n,
        "failure_rate": 100.0 * fails  / n,
        "repair_success_rate": (100.0 * rep_good / rep_total) if rep_total else 0.0,
        "avg_latency_ms": statistics.mean(lat) if lat else 0.0,
        "p95_latency_ms": int(pd.Series(lat).quantile(0.95)) if lat else 0,
    }

def aggregate_from_memory(run_dirs: list[str]):
    """
    Build metrics only from the run dirs provided, using in-memory records.
    Returns:
      - overall_by_model: {model: metrics}
      - by_model_domain: {(model, domain): metrics}
      - by_model_family: {(model, family): metrics}
    """
    per_model = defaultdict(list)
    per_model_domain = defaultdict(list)
    per_model_family = defaultdict(list)

    for rd in run_dirs:
        recs = st.session_state["runs"].get(rd, [])
        for r in recs:
            model = r.get("model") or "unknown_model"
            fam   = r.get("family") or "unknown_family"
            dom   = domain_of_family(fam)
            per_model[model].append(r)
            per_model_domain[(model, dom)].append(r)
            per_model_family[(model, fam)].append(r)

    overall = {m: metrics_from_records(rs) for m, rs in per_model.items()}
    by_domain = {(m,d): metrics_from_records(rs) for (m,d), rs in per_model_domain.items()}
    by_family = {(m,f): metrics_from_records(rs) for (m,f), rs in per_model_family.items()}
    return overall, by_domain, by_family

def df_overall(overall: dict) -> pd.DataFrame:
    df = pd.DataFrame(overall).T.reset_index().rename(columns={"index":"model"})
    if df.empty: return df
    cols = ["model","count","accuracy","failure_rate","repair_success_rate","avg_latency_ms","p95_latency_ms"]
    return df[cols].sort_values("accuracy", ascending=False)

def df_by_domain(by_domain: dict) -> pd.DataFrame:
    rows = []
    for (m,d), mtr in by_domain.items():
        rows.append(dict(model=m, domain=d, **mtr))
    return pd.DataFrame(rows)

def df_by_family(by_family: dict) -> pd.DataFrame:
    rows = []
    for (m,f), mtr in by_family.items():
        rows.append(dict(model=m, family=f, **mtr))
    return pd.DataFrame(rows)
    
def ensure_dirs():
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(SUITES_DIR, exist_ok=True)


def list_run_dirs(base: str = RUNS_DIR) -> List[str]:
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'results.jsonl')):
            out.append(p)
    return out


def list_suites(base: str = SUITES_DIR) -> List[str]:
    ensure_dirs()
    return [os.path.join(base, f) for f in sorted(os.listdir(base)) if f.endswith('.jsonl')]


def load_results_jsonl(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, 'results.jsonl')
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def df_from_results(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    flat_rows = []
    for r in rows:
        rr = dict(r)
        v = rr.get('validators', {}) or {}
        rr['val_json_schema'] = v.get('json_schema')
        rr['val_csv'] = v.get('csv')
        rr['val_html'] = v.get('html')
        rr['strategies_str'] = ", ".join(rr.get('strategies_applied', []) or [])
        expl = (rr.get('explanations', {}) or {}).get('format')
        if isinstance(expl, dict):
            rr['explain_reason'] = expl.get('reason')
            rr['explain_fix'] = expl.get('fix')
        else:
            rr['explain_reason'] = None
            rr['explain_fix'] = None
        flat_rows.append(rr)
    df = pd.DataFrame(flat_rows)
    prefer_cols = [
        'case_id','family','model','final_status','taxonomy','status_before','taxonomy_before',
        'latency_ms','strategies_str','val_json_schema','val_csv','val_html','error'
    ]
    text_cols = ['prompt','output','explain_reason','explain_fix']
    cols = [c for c in prefer_cols if c in df.columns] + [c for c in text_cols if c in df.columns] + [c for c in df.columns if c not in prefer_cols + text_cols]
    return df[cols]


def plot_passrate_by_family(summary: Dict[str, Any], title: str = 'Pass rate by family'):
    byfam = summary.get('by_family', {})
    if not byfam:
        st.info('No by_family metrics.')
        return
    fams, rates, counts = [], [], []
    for fam, obj in byfam.items():
        fams.append(fam)
        rates.append(obj.get('pass_rate', 0))
        counts.append(obj.get('count', 0))
    fig = px.bar(x=fams, y=rates, title=title, labels={'x':'Family','y':'Pass rate'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Counts: " + ", ".join(f"{f}={c}" for f,c in zip(fams, counts)))


def plot_taxonomy(summary: Dict[str, Any], title: str = 'Failure taxonomy'):
    tax = summary.get('taxonomy', {}) or {}
    if not tax:
        st.info('No failures recorded (nice!).')
        return
    labels = list(tax.keys())
    values = list(tax.values())
    fig = px.pie(names=labels, values=values, title=title)
    st.plotly_chart(fig, use_container_width=True)


def plot_strategy_effectiveness(summary: Dict[str, Any]):
    se = summary.get('strategy_effectiveness', {}) or {}
    fixed = se.get('fixed_by_strategy', {}) or {}
    if not fixed:
        st.info('No strategies were applied or no flips recorded.')
        return
    labels = list(fixed.keys())
    values = list(fixed.values())
    fig = px.bar(x=labels, y=values, title='Failures fixed by strategy', labels={'x':'Strategy','y':'# fixed'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Total flips fail→pass: {se.get('flips_fail_to_pass', 0)}")


def render_run_summary(run_dir: str):
    st.subheader('Run Summary')
    summary = summarize(run_dir)
    col1, col2 = st.columns(2)
    with col1:
        plot_passrate_by_family(summary)
        cA, cB = st.columns(2)
        with cA:
            kpi('Latency p50 (ms)', str(summary['latency'].get('p50') or '—'))
        with cB:
            kpi('Latency p95 (ms)', str(summary['latency'].get('p95') or '—'))
    with col2:
        plot_taxonomy(summary)
        plot_strategy_effectiveness(summary)


def render_diff(off_dir: str, on_dir: str):
    st.subheader('Ablation: OFF vs ON')
    d = diff_runs(off_dir, on_dir)
    rows = []
    for fam, obj in d['by_family'].items():
        rows.append({'family': fam, 'off': obj['off'], 'on': obj['on'], 'delta': obj['delta']})
    df = pd.DataFrame(rows).sort_values('family')
    st.dataframe(df, use_container_width=True)
    fig = px.bar(df, x='family', y='delta', title='Pass rate delta (ON - OFF)', labels={'delta':'Δ pass rate'})
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write('**Failures**')
        st.json(d['taxonomy_shift'])
    with c2:
        st.write('**Latency**')
        st.json(d['latency_shift'])
    st.write('**Strategy effectiveness (ON)**')
    st.json(d.get('on_strategy_effectiveness', {}))


def render_drilldown(run_dir: str):
    st.subheader('Drill-down (cases)')
    rows = load_results_jsonl(run_dir)
    df = df_from_results(rows)
    if df.empty:
        st.info('No results found.')
        return
    families = sorted([x for x in df['family'].dropna().unique()])
    models = sorted([x for x in df['model'].dropna().unique()])
    statuses = ['pass','fail']
    fam_sel = st.multiselect('Family', families, default=families)
    model_sel = st.multiselect('Model', models, default=models)
    status_sel = st.multiselect('Final status', statuses, default=statuses)
    q = st.text_input('Search in prompt/output (substring)')
    m = df[df['family'].isin(fam_sel) & df['model'].isin(model_sel) & df['final_status'].isin(status_sel)]
    if q:
        ql = q.lower()
        def _match(s: str) -> bool:
            if not isinstance(s, str):
                return False
            return ql in s.lower()
        mask = m['prompt'].apply(_match) | m['output'].apply(_match)
        m = m[mask]
    st.caption(f"Showing {len(m)} / {len(df)} records")
    st.dataframe(m, use_container_width=True, height=420)
    st.write('---')
    st.write('**Case focus**')
    opt_case = st.selectbox('Pick a case_id', options=sorted(m['case_id'].dropna().unique()))
    row = m[m['case_id'] == opt_case].iloc[0].to_dict()
    colA, colB = st.columns(2)
    with colA:
        st.write('**Prompt**')
        st.code(row.get('prompt') or '', language='text')
        st.write('**Output**')
        st.code(row.get('output') or '', language='json' if row.get('family')=='json_schema' else 'text')
    with colB:
        st.write('**Validators**')
        val = {k:v for k,v in row.items() if k.startswith('val_')}
        st.json(val)
        st.write('**Status & Taxonomy**')
        st.json({'status_before': row.get('status_before'), 'taxonomy_before': row.get('taxonomy_before'), 'final_status': row.get('final_status'), 'taxonomy': row.get('taxonomy')})
        if row.get('explain_reason') or row.get('explain_fix'):
            st.write('**AI explanation**')
            st.json({'reason': row.get('explain_reason'), 'fix': row.get('explain_fix')})


# ---------- Generators (simple, local) ----------
# Minimal built-in generators here to avoid extra imports. You may replace with your generator/ modules.

def gen_json_cases(n: int = 1) -> List[Dict[str, Any]]:
    cases = []
    schema = {"type":"object","required":["name","age"],"properties":{"name":{"type":"string"},"age":{"type":"number"}},"additionalProperties":True}
    for i in range(n):
        prompt = "Return ONLY JSON with fields {\"name\": string, \"age\": number}."
        if i % 3 == 1:
            prompt = "Respond ONLY with JSON: {\"name\": \"Alice\", \"age\": 30}. No extra text."
        elif i % 3 == 2:
            prompt = "Output a JSON object with keys name (string) and age (number). Do not include explanations or code fences."
        cases.append({
            "id": f"json_{i:03d}",
            "family": "json_schema",
            "prompt": prompt,
            "expect": {"type":"json","schema": schema}
        })
    return cases


def gen_csv_cases(n: int = 1) -> List[Dict[str, Any]]:
    cases = []
    cols = ["name","amount","currency"]
    for i in range(n):
        prompt = "Output ONLY CSV with header name,amount,currency and exactly 3 data rows. No code fences, no commentary."
        if i % 3 == 1:
            prompt = "ONLY CSV with header name,amount,currency and exactly 3 rows. Do not add explanations."
        elif i % 3 == 2:
            prompt = "Return ONLY CSV header name,amount,currency and 3 rows. Do not include markdown code fences."
        cases.append({
            "id": f"csv_{i:03d}",
            "family": "csv_strict",
            "prompt": prompt,
            "expect": {"type":"csv","columns": cols, "rows": 3, "strict_no_extra_text": True}
        })
    return cases


def gen_html_cases(n: int = 1) -> List[Dict[str, Any]]:
    cases = []
    for i in range(n):
        prompt = "Return ONLY a valid minimal HTML snippet containing exactly one <title>, one <h1>, and one <p>. Do not include code fences or explanations."
        if i % 3 == 1:
            prompt = "ONLY minimal HTML snippet with one <title>, one <h1>, one <p>. No extra text."
        elif i % 3 == 2:
            prompt = "Return ONLY minimal HTML snippet (exactly one <title>, <h1>, <p>). Do not wrap in ``` fences."
        cases.append({
            "id": f"html_{i:03d}",
            "family": "html_snippet",
            "prompt": prompt,
            "expect": {"type":"html","required_tags":["title","h1","p"], "strict_no_extra_text": True}
        })
    return cases


def write_suite(path: str, cases: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# ---------- Run helpers ----------

def build_adapters_from_ui(provider: str, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    key = f"{provider}_{model_name}"
    if provider == 'openai':
        adapter = OpenAIAdapter(model_name, **params)
    elif provider == 'huggingface':
        # Never pass force_json default; runner will set it per-case if needed
        _params = dict(params)
        _params.pop("force_json", None)
        adapter = HFAdapter(model_name, **_params)

        device = next(adapter.model.parameters()).device
        st.write(f"🤖 Model loaded on device: {device}")
    else:
        raise ValueError('Unsupported provider')
    return {key: adapter}



def run_experiment_sync(cases: List[Dict[str, Any]], adapters: Dict[str, Any], run_dir: str, cfg: Dict[str, Any]):
    import asyncio
    os.makedirs(run_dir, exist_ok=True)
    asyncio.run(run_experiment(cases, adapters, run_dir, cfg))


# ---------- PAGES ----------

st.set_page_config(page_title='FailProof LLM', layout='wide', page_icon='✨')
ensure_dirs()
apply_theme()

st.sidebar.title('FailProof LLM')
st.sidebar.markdown("<span class='chip'>LLM Reliability · Minimal UI</span>", unsafe_allow_html=True)
page = st.sidebar.radio('Navigate', ['Home','Dataset Studio','Run & Analyze'])

# ---- Home ----
if page == 'Home':
    hero('FailProof LLM — Stress‑test AI with Edge Cases', 'Generate adversarial suites → run across models → analyze, explain, and harden.')
    st.markdown('     ')

    
    import streamlit as st

    st.markdown("""
    
    ---
    
    ## **Problem Statement**
    Large Language Models (LLMs) are powerful but vulnerable to adversarial prompts, formatting stress tests, and domain-specific edge cases.  
    These vulnerabilities can cause LLMs to:
    - Produce incorrect or unsafe outputs
    - Fail to comply with structured output requirements
    - Break under edge-case inputs  
    
    There is currently no **practical, end-to-end benchmark** for developers to evaluate and compare models across *both* adversarial robustness and domain accuracy.
    
    ---
    
    ## **Our Solution**
    We built **FailProof-LLM** — a unified platform for:
    1. **Synthetic Dataset Generation**  
       Prompt-driven creation of high-quality test cases across multiple robustness categories and domains.
    2. **Multi-Model Evaluation**  
       Compare models side-by-side on structured, adversarial, and domain-specific tasks.
    3. **Automated Validation & Repair**  
       Validate outputs using JSON schema, CSV header checks, HTML parsing, etc., and attempt automatic fixes.
    4. **Interactive Analytics**  
       Real-time benchmarking dashboard with domain/category-wise breakdowns and model comparisons.
    
    ---
    
    ## **Instructions to Use the Platform**
    1. **Start at the Home Page** → Pick your workflow: *Dataset Studio* or *Run & Analyze*.
    2. **Generate Datasets** → In *Dataset Studio*, select a category/domain, choose dataset size, and generate cases.
    3. **Run Evaluation** → In *Run & Analyze*, choose your dataset and the models to test.
    4. **View Results** → Explore pass/fail breakdowns, repair success, latency, and model-wise comparisons.
    5. **Benchmark** → See aggregated metrics per category/domain to understand model strengths and weaknesses.
    
    ---
    
    ## **Models We Considered**
    - **GPT-4o-mini** (OpenAI) — High-accuracy, robust structured output
    - **Qwen1.5-0.5B** (Alibaba) — Strong mid-range performance
    - **TinyLlama-1.1B-Chat** — Lightweight, low-compute option
    - **FLAN-T5-base** & **FLAN-T5-small** (Google) — Open-source baselines
    
    ---
    
    ## **Domains & Adversarial Categories**
    We evaluate across **3 domains** and **22 adversarial robustness categories**.
    
    **Domains:**
    - Financial
    - Healthcare
    - Legal
    
    **Adversarial Categories:**
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
       - Scenarios relevant to financial, legal, or healthcare rules
    
    ---
    
    ## **Evaluation Metrics**
    - **Accuracy** — % of cases passing validation  
    - **Validation Pass Rate** — Domain-specific validator success  
    - **Repair Success Rate** — % of failed cases fixed by auto-repair  
    - **Failure Rate** — % still failing after repair  
    - **Latency (ms)** — Avg. generation time  
    - **Token Efficiency** — Passes per token generated  
    
    These metrics are shown per **category/domain** with **model-wise comparisons** in the dashboard.
    
    ---
    """)
    st.subheader("Benchmark — Rows: Domains & Categories • Columns: Metrics (one box per model)")

    table_df = build_multiindex_table(metrics_data)
    st.dataframe(
        style_multiindex(table_df),
        use_container_width=True,
        height=min(700, 100 + 28 * len(table_df)),  # auto-ish height
    )
    st.caption("Tip: scroll horizontally to compare models per metric; columns are grouped by metric, each with sub-columns for models.")
# ---- Dataset Studio ----
elif page == 'Dataset Studio':
    hero('Dataset Studio', 'Generate validator-ready suites from rich templates.')
    st.markdown('     ')
    st.caption("Pick a category template, choose size, and generate a JSONL suite you can immediately run.")

    # Discover templates on the fly
    # Discover templates
    templates = discover_templates("templates")
    domain_opts = ["All", "Healthcare", "Legal", "Financial"]
    sel_domain = st.selectbox("Domain", domain_opts, index=0)
    
    if not templates:
        st.warning("No templates found under ./templates. Add .md files (category prompts) to proceed.")
        st.stop()
    
    def _domain_of(t):
        p = t["path"].lower()
        if "/domains/healthcare/" in p: return "Healthcare"
        if "/domains/legal/" in p: return "Legal"
        if "/domains/financial/" in p: return "Financial"
        return "All"
    
    # Apply domain filter
    filtered = [t for t in templates if sel_domain == "All" or _domain_of(t) == sel_domain]
    
    # Build selection list (only from filtered)
    opts = {f"{t['group']} / {t['key']}": t for t in filtered}
    sel = st.selectbox("Category template", options=list(opts.keys()))
    
    # Selected template object
    t = opts[sel]

    # Let user edit friendly labels
    colA, colB = st.columns([0.6, 0.4])
    with colA:
        family = st.text_input("Family key", value=t["key"])
        category_label = st.text_input("Category label", value=t["label"])
    with colB:
        total_items = st.number_input("Dataset size", min_value=1, max_value=2000, value=50, step=10)
        chunk_size = st.slider("Chunk size (per model call)", min_value=10, max_value=200, value=50, step=10)

    col3, col4, col5 = st.columns([0.35,0.35,0.30])
    with col3:
        model_name = st.text_input("Generation model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    with col4:
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=1234, step=1)
    with col5:
        out_name = st.text_input("Output JSONL", value=f"data/suites/{family}_{total_items}.jsonl")

    st.write("")
    go = st.button("Generate dataset", use_container_width=True)
    if go:
        try:
            with st.spinner("Generating dataset from template..."):
                openai.api_key = os.getenv("OPENAI_API_KEY")  # or from secrets
                if total_items <= chunk_size:
                    rows = generate_from_template(
                        template_path=t["path"],
                        out_path=out_name,
                        family=family,
                        category_label=category_label,
                        num_items=int(total_items),
                        seed=int(seed),
                        model=model_name,
                        temperature=float(temperature),
                    )
                else:
                    rows = generate_from_template_chunked(
                        template_path=t["path"],
                        out_path=out_name,
                        family=family,
                        category_label=category_label,
                        total_items=int(total_items),
                        chunk_size=int(chunk_size),
                        seed=int(seed),
                        model=model_name,
                        temperature=float(temperature),
                    )
            st.success(f"Done! Wrote {len(rows)} cases → {out_name}")
            st.session_state["last_suite"] = out_name
            st.code("\n".join([json.dumps(r, ensure_ascii=False) for r in rows[:5]]), language="json")
        except Exception as e:
            st.error(f"Generation failed: {e}")


# ---- Run & Analyze ----
else:
    hero('Run & Analyze', 'Select a suite and a model. Toggle guards. Inspect outcomes & ablations.')
    st.markdown('     ')

    # Controls
    suites = list_suites()
    if not suites:
        st.warning('No suites found. Go to **Dataset Studio** to generate one.')
        st.stop()

    # Prefer last generated suite if present
    default_suite_idx = len(suites) - 1
    last_suite = st.session_state.get("last_suite")
    if last_suite and last_suite in suites:
        default_suite_idx = suites.index(last_suite)

    sel_suite = st.selectbox('Suite file', options=suites, index=max(0, default_suite_idx))

    model_provider = st.selectbox("Provider", ["openai", "huggingface"], index=0)
    if model_provider == "huggingface":
        model_name = st.selectbox("HF model", HF_PRESETS, index=0)
    else:
        model_name = st.text_input("OpenAI model", value="gpt-4o-mini")

    st.write("HuggingFace models might now work as I don't have a server to host them. But, all the results are obtained through analysis done in the local machine")
    temperature = st.slider('Temperature', 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.number_input('Max tokens', 32, 4096, 512, 32)

    st.markdown('**Guards**')
    colg1, colg2 = st.columns(2)
    with colg1:
        repairs = st.checkbox('Enable repairs (JSON/CSV/HTML)', value=True)
    with colg2:
        sc_json = st.number_input('Self-consistency for JSON (N)', 1, 5, 1, 1)

    colr1, colr2 = st.columns(2)
    with colr1:
        do_off = st.button('Run OFF (repairs off)')
    with colr2:
        do_on = st.button('Run ON (repairs on)')

    # Build adapters & cases
    # Load cases from suite
    cases: List[Dict[str, Any]] = []
    with open(sel_suite, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    params = {"temperature": float(temperature), "max_tokens": int(max_tokens)}
    adapters = build_adapters_from_ui(model_provider, model_name, params)

    

    # Run buttons
    if do_off:
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = os.path.join(RUNS_DIR, f"OFF-{ts}")
        cfg = {
            "suite": sel_suite,
            "models": [{"key": list(adapters.keys())[0], "provider": model_provider, "model": model_name, "params": params}],
            "guards": {"repairs": False, "self_consistency_json": 1},
            "critics": {"format_explainer": {"enabled": True, "model": model_name, "max_tokens": 120}},
            "run": {"parallelism": 8, "timeout_s": 20, "retries": 1}
        }
        with st.spinner('Running OFF (repairs disabled)...'):
            run_experiment_sync(cases, adapters, run_dir, cfg)
        st.success(f"OFF run complete → {run_dir}")
        remember_run(run_dir)

    if do_on:
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = os.path.join(RUNS_DIR, f"ON-{ts}")
        cfg = {
            "suite": sel_suite,
            "models": [{"key": list(adapters.keys())[0], "provider": model_provider, "model": model_name, "params": params}],
            "guards": {"repairs": True, "self_consistency_json": int(sc_json)},
            "critics": {"format_explainer": {"enabled": True, "model": model_name, "max_tokens": 120}},
            "run": {"parallelism": 8, "timeout_s": 20, "retries": 1}
        }
        with st.spinner('Running ON (repairs enabled)...'):
            run_experiment_sync(cases, adapters, run_dir, cfg)
        st.success(f"ON run complete → {run_dir}")
        remember_run(run_dir)

    st.write('---')

    # Analysis tabs
    run_dirs = list_run_dirs()
    if not run_dirs:
        st.info('No completed runs yet.')
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(['Run Summary', 'Ablation (OFF vs ON)', 'Drill-down', 'Deeper Analysis'])

    with tab1:
        sel_run = st.selectbox('Select run dir', options=run_dirs, index=len(run_dirs)-1, key='rs')
        render_run_summary(sel_run)

    with tab2:
        # Try to select the last OFF and last ON if present
        offs = [r for r in run_dirs if os.path.basename(r).startswith('OFF-')]
        ons  = [r for r in run_dirs if os.path.basename(r).startswith('ON-')]
        off_run = st.selectbox('OFF run dir', options=(offs or run_dirs), index=max(0, len(offs)-1), key='offsel')
        on_run  = st.selectbox('ON run dir',  options=(ons or run_dirs),  index=max(0, len(ons)-1),  key='onsel')
        if off_run == on_run:
            st.info('Select two different runs to view the diff.')
        else:
            render_diff(off_run, on_run)

    with tab3:
        sel_run_for_drill = st.selectbox('Run dir for drill-down', options=run_dirs, index=len(run_dirs)-1, key='dr')
        render_drilldown(sel_run_for_drill)

    with tab4:
        st.subheader("Analysis (current selection)")
    
        # Use *selected* run dirs from your two dropdowns in Ablation (or add selectors here)
        selected_dirs = []
        try:
            selected_dirs.extend([off_sel, on_sel])  # if you have off_sel/on_sel in scope
        except NameError:
            pass
    
        # Fallback: any runs already cached
        if not selected_dirs:
            selected_dirs = list(st.session_state["runs"].keys())
    
        selected_dirs = [d for d in selected_dirs if d in st.session_state["runs"]]
        if not selected_dirs:
            st.info("No runs loaded in memory yet. Run a suite or load a run to analyze.")
            st.stop()
    
        overall, by_domain, by_family = aggregate_from_memory(selected_dirs)
    
        st.markdown("**Overall (per model)**")
        dfO = df_overall(overall)
        if not dfO.empty:
            st.dataframe(dfO.style.format({
                "accuracy":"{:.2f}", "failure_rate":"{:.2f}",
                "repair_success_rate":"{:.2f}",
                "avg_latency_ms":"{:.0f}", "p95_latency_ms":"{:.0f}"}), use_container_width=True)
    
        st.markdown("**By Domain**")
        dfD = df_by_domain(by_domain)
        if not dfD.empty:
            st.dataframe(dfD.style.format({
                "accuracy":"{:.2f}", "failure_rate":"{:.2f}",
                "repair_success_rate":"{:.2f}",
                "avg_latency_ms":"{:.0f}", "p95_latency_ms":"{:.0f}"}), use_container_width=True)
            st.plotly_chart(px.bar(dfD, x="domain", y="accuracy", color="model",
                                   barmode="group", title="Accuracy by Domain"),
                            use_container_width=True)
    
        st.markdown("**By Category / Family**")
        dfF = df_by_family(by_family)
        if not dfF.empty:
            st.dataframe(dfF.style.format({
                "accuracy":"{:.2f}", "failure_rate":"{:.2f}",
                "repair_success_rate":"{:.2f}",
                "avg_latency_ms":"{:.0f}", "p95_latency_ms":"{:.0f}"}), use_container_width=True)
            # Optional: show top families by volume for clarity
            top_fams = dfF.groupby("family")["count"].sum().sort_values(ascending=False).head(12).index.tolist()
            dfTop = dfF[dfF["family"].isin(top_fams)]
            st.plotly_chart(px.bar(dfTop, x="family", y="accuracy", color="model",
                                   barmode="group", title="Top Families by Accuracy"),
                            use_container_width=True)
    
