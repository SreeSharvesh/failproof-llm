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

    
    st.markdown(
        """
        **Workflow**
        1. **Dataset Studio** → Pick categories (JSON, CSV, HTML), generate a suite.
        2. **Run & Analyze** → Select model + guards, run tests, view metrics & drill-down.
        3. (Optional) Re-run with guards OFF vs ON to see measurable gains.
        4. (Optional) Generate **harder follow-ups** from failures.
        """
    )
    st.info('Tip: Start in **Dataset Studio** to create your suite, then head to **Run & Analyze**.')

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

    st.write('---')

    # Analysis tabs
    run_dirs = list_run_dirs()
    if not run_dirs:
        st.info('No completed runs yet.')
        st.stop()

    tab1, tab2, tab3 = st.tabs(['Run Summary', 'Ablation (OFF vs ON)', 'Drill-down'])

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
