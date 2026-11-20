import os
import html
import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
CHURN_VALUES = ["yes", "true", "1", "churned"]
COX_CANDIDATES = [
    "tenure",
    "monthly_charges",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Contract",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaymentMethod",
]

# -------------------- Database --------------------
@st.cache_data(ttl=300)
def load_customers():
    config = dict(st.secrets["postgres"])
    if "database" in config:
        config["dbname"] = config.pop("database")

    with psycopg.connect(**config) as conn:
        df = pd.read_sql_query("SELECT * FROM telco_customers", conn)

    df.columns = [str(col).strip().lower() for col in df.columns]

    if "monthlycharges" in df.columns:
        df = df.rename(columns={"monthlycharges": "monthly_charges"})
    if "totalcharges" in df.columns:
        df = df.rename(columns={"totalcharges": "total_charges"})
    if "customerid" in df.columns:
        df = df.rename(columns={"customerid": "customer_id"})

    return df

def load_csv(filename, warn=True):
    # Analysis files are organised inside the outputs/ folder.
    # The other paths are kept as fallbacks for local compatibility.
    possible_paths = [
        APP_DIR / "outputs" / filename,
        APP_DIR / "csv files" / filename,
        APP_DIR / filename,
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as error:
                st.warning(f"Could not read {filename}: {error}")
                return None

    if warn:
        st.warning(f"CSV not found in the outputs folder: {filename}")
    return None

def find_col(df, candidates):
    if df is None or df.empty:
        return None

    columns = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in columns:
            return columns[key]

    return None

def is_churn(series):
    return series.astype(str).str.strip().str.lower().isin(CHURN_VALUES)

def money(value):
    return f"${value:,.2f}"

def money0(value):
    return f"${value:,.0f}"


# -------------------- Small display helpers --------------------
def safe(text):
    """Stop Streamlit's markdown from treating $ signs as maths."""
    return text.replace("$", "&#36;")

def b(text):
    return f"<strong>{html.escape(str(text))}</strong>"

def insight(text):
    """One-line takeaway shown under a chart or table."""
    st.markdown(
        f'<div class="insight"><span class="insight-tag">So what</span>'
        f"<span>{safe(text)}</span></div>",
        unsafe_allow_html=True,
    )

ACRONYMS = {"id": "ID", "clv": "CLV", "roi": "ROI", "shap": "SHAP", "hr": "HR"}
MONEY_WORDS = ("charge", "revenue", "clv", "cost", "benefit", "saving", "preserved")
PCT_WORDS = ("rate", "probability", "prob", "percent", "pct", "share", "roi")

def pretty_label(name):
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name).replace("_", " "))
    words = text.split()
    out = []
    for i, word in enumerate(words):
        if word.lower() in ACRONYMS:
            out.append(ACRONYMS[word.lower()])
        elif i == 0:
            out.append(word.capitalize())
        else:
            out.append(word.lower())
    return " ".join(out)

def column_kind(name, series):
    lname = str(name).lower()
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "text"
    if "p-value" in lname:
        return "pvalue"
    if any(word in lname for word in MONEY_WORDS):
        return "money"
    if any(word in lname for word in PCT_WORDS):
        return "pct"
    clean = series.dropna()
    if not clean.empty and (clean % 1 == 0).all():
        return "int"
    return "num"

def format_value(value, kind, decimals=2, pct_scale=1):
    if pd.isna(value):
        return "—"
    if kind == "money":
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value):,.2f}"
    if kind == "pct":
        return f"{value * pct_scale:.1f}%"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "pvalue":
        return "<0.0001" if value < 0.0001 else f"{value:.4f}"
    if kind == "num":
        return f"{value:,.{decimals}f}"
    return str(value)

def render_table(df, max_rows=300, height=420, badge_col=None, decimals=2, rename=None):
    """Theme-aware HTML table (works in both Light and Dark mode)."""
    if df is None or df.empty:
        st.info("No rows to show.")
        return

    rename = rename or {}
    shown = df.head(max_rows)
    kinds, scales = {}, {}
    for col in shown.columns:
        kinds[col] = column_kind(col, df[col])
        if kinds[col] == "pct" and df[col].abs().max() <= 1:
            scales[col] = 100
        else:
            scales[col] = 1

    head = "".join(
        f'<th class="{"num" if kinds[col] != "text" else ""}">'
        f"{html.escape(rename.get(col, pretty_label(col)))}</th>"
        for col in shown.columns
    )

    badge_classes = {
        "high": "badge-high",
        "medium": "badge-medium",
        "low": "badge-low",
    }
    rows = []
    for _, row in shown.iterrows():
        cells = []
        for col in shown.columns:
            text = html.escape(
                format_value(row[col], kinds[col], decimals, scales[col])
            )
            if badge_col is not None and col == badge_col:
                css = badge_classes.get(str(row[col]).strip().lower())
                if css:
                    text = f'<span class="badge {css}">{text}</span>'
            align = "num" if kinds[col] != "text" else ""
            cells.append(f'<td class="{align}">{text}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        f'<div class="table-wrap" style="max-height:{height}px">'
        f'<table class="data-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )
    st.markdown(safe(table_html), unsafe_allow_html=True)


# -------------------- Cached heavy analysis --------------------
@st.cache_data(show_spinner=False)
def run_survival(df, tenure_col, churn_col, contract_col):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    base = df.copy()
    base["_t"] = pd.to_numeric(base[tenure_col], errors="coerce")
    base = base.dropna(subset=["_t"])
    base = base[base["_t"] >= 0]
    base["_e"] = is_churn(base[churn_col]).astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(durations=base["_t"], event_observed=base["_e"], label="All customers")

    curve = kmf.survival_function_.reset_index()
    curve.columns = ["Tenure (months)", "Churn-free probability"]

    result = {
        "curve": curve,
        "s12": float(kmf.predict(12)),
        "s24": float(kmf.predict(24)),
        "contract_long": None,
        "contract_s24": {},
        "p_value": None,
        "p_error": None,
    }

    if contract_col:
        grouped = base.dropna(subset=[contract_col])
        frames = []
        for name, group in grouped.groupby(contract_col):
            group_kmf = KaplanMeierFitter()
            group_kmf.fit(
                durations=group["_t"],
                event_observed=group["_e"],
                label=str(name),
            )
            group_curve = group_kmf.survival_function_.reset_index()
            group_curve.columns = ["Tenure (months)", str(name)]
            frames.append(group_curve)
            result["contract_s24"][str(name)] = float(group_kmf.predict(24))

        if frames:
            combined = frames[0]
            for frame in frames[1:]:
                combined = combined.merge(frame, on="Tenure (months)", how="outer")
            result["contract_long"] = combined.melt(
                id_vars=["Tenure (months)"],
                var_name="Contract",
                value_name="Churn-free probability",
            ).dropna()

        try:
            logrank = multivariate_logrank_test(
                event_durations=grouped["_t"],
                groups=grouped[contract_col],
                event_observed=grouped["_e"],
            )
            result["p_value"] = float(logrank.p_value)
        except Exception as error:
            result["p_error"] = str(error)

    return result

@st.cache_data(show_spinner=False)
def run_cox(df, tenure_col, churn_col):
    from lifelines import CoxPHFitter

    cox_df = df.copy()
    cox_df["_duration"] = pd.to_numeric(cox_df[tenure_col], errors="coerce")
    cox_df["_event"] = is_churn(cox_df[churn_col]).astype(int)

    feature_cols = [
        col for col in COX_CANDIDATES
        if col in cox_df.columns and col != tenure_col
    ]
    if not feature_cols:
        return {
            "results": None,
            "warning": "No eligible customer features were found for the Cox model.",
        }

    model_df = cox_df[["_duration", "_event"] + feature_cols].copy()
    model_df = model_df.dropna(subset=["_duration", "_event"])
    model_df = model_df[model_df["_duration"] > 0]

    for col in feature_cols:
        if model_df[col].dtype == "object":
            model_df[col] = model_df[col].fillna("Unknown")
        else:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna()

    # One-hot encode categorical features and remove one reference category.
    model_df = pd.get_dummies(
        model_df,
        columns=[col for col in feature_cols if model_df[col].dtype == "object"],
        drop_first=True,
        dtype=float,
    )

    # Remove constant columns, which cannot be estimated.
    predictors = [
        col for col in model_df.columns
        if col not in ["_duration", "_event"] and model_df[col].nunique() > 1
    ]
    model_df = model_df[["_duration", "_event"] + predictors]

    if not predictors:
        return {"results": None, "warning": "The selected features have no variation."}

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(model_df, duration_col="_duration", event_col="_event")

    # Check the proportional-hazards assumption (a violation does not stop the
    # model from running, so it is reported as a warning rather than an error).
    ph_flagged, ph_error = [], None
    try:
        from lifelines.statistics import proportional_hazard_test

        ph_summary = proportional_hazard_test(
            cph, model_df, time_transform="rank"
        ).summary
        ph_flagged = [str(name) for name in ph_summary.index[ph_summary["p"] < 0.05]]
    except Exception as error:
        ph_error = str(error)

    results = cph.summary.reset_index().rename(
        columns={
            "covariate": "Feature",
            "exp(coef)": "Hazard ratio",
            "exp(coef) lower 95%": "HR 95% lower",
            "exp(coef) upper 95%": "HR 95% upper",
            "p": "p-value",
        }
    )
    return {
        "results": results,
        "warning": None,
        "ph_flagged": ph_flagged,
        "ph_error": ph_error,
    }

# -------------------- Load data --------------------
db_connected = False
try:
    customers_df = load_customers()
    db_connected = True
except Exception as error:
    st.error(f"Could not load PostgreSQL customer records: {error}")
    st.stop()

if customers_df.empty:
    st.warning("The PostgreSQL table has no customer records.")
    st.stop()

churn_col = find_col(customers_df, ["churn"])
monthly_col = find_col(customers_df, ["monthly_charges"])
tenure_col = find_col(customers_df, ["tenure"])
contract_col = find_col(customers_df, ["contract"])
customer_id_col = find_col(customers_df, ["customer_id"])

if churn_col is None:
    st.error("Could not find the churn column.")
    st.stop()

customers_df["_churned"] = is_churn(customers_df[churn_col])
if customer_id_col:
    customers_df[customer_id_col] = customers_df[customer_id_col].astype(str).str.strip()

# Tidy customer IDs (stray spaces) so they can be matched against the risk CSV.
if customer_id_col:
    customers_df[customer_id_col] = customers_df[customer_id_col].astype(str).str.strip()

total_customers = len(customers_df)
churned_customers = int(customers_df["_churned"].sum())
churn_rate = churned_customers / total_customers

if monthly_col:
    customers_df[monthly_col] = pd.to_numeric(
        customers_df[monthly_col], errors="coerce"
    )
    avg_monthly_charge = float(customers_df[monthly_col].mean())
    monthly_revenue = float(customers_df[monthly_col].sum())
else:
    avg_monthly_charge = 0.0
    monthly_revenue = 0.0

if tenure_col:
    customers_df[tenure_col] = pd.to_numeric(
        customers_df[tenure_col], errors="coerce"
    )

# -------------------- Saved analysis files --------------------
shap_df = load_csv("shap_feature_importance.csv")
cluster_df = load_csv("customer_cluster_profile.csv")
clv_df = load_csv("clv_summary.csv")
contract_df = load_csv("contract_retention_summary.csv")
roi_scenario_df = load_csv("retention_roi_scenario.csv")

# Customer-level churn risk (loaded quietly; the Take action page explains if missing)
risk_raw = load_csv("customer_churn_risk.csv", warn=False)
risk = None
risk_error = False
if risk_raw is not None:
    _r = risk_raw.copy()
    _r.columns = [str(col).strip().lower() for col in _r.columns]
    _id = find_col(_r, ["customerid", "customer_id", "customer"])
    _prob = find_col(_r, ["churn_probability", "churn_prob", "probability"])
    _level = find_col(_r, ["risk_level"])
    _focus = find_col(_r, ["contact_focus"])
    if any(col is None for col in [_id, _prob, _level, _focus]):
        risk_error = True
    else:
        _r[_prob] = pd.to_numeric(_r[_prob], errors="coerce")
        _r[_id] = _r[_id].astype(str).str.strip()
        _r = _r.dropna(subset=[_prob]).sort_values(_prob, ascending=False)
        _r[_id] = _r[_id].astype(str).str.strip()
        risk = {"df": _r, "id": _id, "prob": _prob, "level": _level, "focus": _focus}

        # Check that the risk file's customer IDs really match the database.
        risk["match"] = None
        if customer_id_col:
            db_ids = set(customers_df[customer_id_col].astype(str).str.strip())
            risk_ids = set(_r[_id])
            risk["match"] = {
                "risk_total": len(risk_ids),
                "risk_matched": len(risk_ids & db_ids),
                "db_total": len(db_ids),
                "db_covered": len(db_ids & risk_ids),
            }

# -------------------- Top bar: brand + theme switch --------------------
header_left, header_right = st.columns([5, 1.6], vertical_alignment="center")

with header_right:
    theme = st.radio(
        "Theme",
        ["Light", "Dark"],
        horizontal=True,
        index=0,
        key="dashboard_theme",
        label_visibility="collapsed",
        format_func=lambda option: "☀️ Light" if option == "Light" else "🌙 Dark",
    )

if theme == "Dark":
    palette = {
        "scheme": "dark",
        "app": "#090F1F",
        "surface": "#111B2E",
        "surface_alt": "#192943",
        "border": "#2B3B56",
        "text": "#F1F5FC",
        "muted": "#A7B5CA",
        "accent": "#A78BFA",
        "accent2": "#38BDF8",
        "on_accent": "#0B1020",
        "shadow": "rgba(3,10,25,.40)",
        "input": "#182338",
        "stay": "#38BDF8",
        "churn": "#FB9B6D",
        "gold": "#FCD34D",
        "cat": ["#A78BFA", "#94A3B8", "#C4B5FD", "#6D5BC6"],
        "high": "#FDBA74",
        "med": "#FCD34D",
        "low": "#6EE7B7",
    }
else:
    palette = {
        "scheme": "light",
        "app": "#F2F5FB",
        "surface": "#FFFFFF",
        "surface_alt": "#F7F8FF",
        "border": "#E3E8F2",
        "text": "#17233B",
        "muted": "#66758F",
        "accent": "#6546D7",
        "accent2": "#168EAD",
        "on_accent": "#FFFFFF",
        "shadow": "rgba(31,48,78,.09)",
        "input": "#FFFFFF",
        "stay": "#168EAD",
        "churn": "#F08A5D",
        "gold": "#D5A441",
        "cat": ["#6546D7", "#64748B", "#A99BEA", "#3A9D8F"],
        "high": "#C2410C",
        "med": "#92660B",
        "low": "#0F7B63",
    }

STAY_COLOR = palette["stay"]
CHURN_COLOR = palette["churn"]
GOLD_COLOR = palette["gold"]
CAT_COLORS = palette["cat"]
ACCENT_COLOR = palette["accent"]

contract_names = (
    sorted(customers_df[contract_col].dropna().astype(str).unique().tolist())
    if contract_col else []
)
contract_color_map = {
    name: CAT_COLORS[i % len(CAT_COLORS)] for i, name in enumerate(contract_names)
}
contract_color_map["Unknown"] = "#94A3B8"

# One colour meaning per concept: teal = stayed, orange = churned,
# purple / grey = categories such as contract type.

ROOT_VARS = f"""
:root {{
    --app-bg: {palette["app"]};
    --surface: {palette["surface"]};
    --surface-alt: {palette["surface_alt"]};
    --border: {palette["border"]};
    --text: {palette["text"]};
    --muted: {palette["muted"]};
    --accent: {palette["accent"]};
    --accent-2: {palette["accent2"]};
    --on-accent: {palette["on_accent"]};
    --shadow: {palette["shadow"]};
    --input-bg: {palette["input"]};
    --high: {palette["high"]};
    --med: {palette["med"]};
    --low: {palette["low"]};
    --gold: {palette["gold"]};
    color-scheme: {palette["scheme"]};
}}
"""

BASE_CSS = """
/* ---------- App shell ---------- */
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stBottom"], section.main {
    background: var(--app-bg) !important;
    color: var(--text);
}
/* Streamlit's transparent top bar sits over the top of the page and blocks clicks
   (it was covering the Light/Dark switch). Make it click-through and leave room for it. */
[data-testid="stHeader"] {
    background: transparent !important;
    pointer-events: none !important;
    height: 2.5rem !important;
}
[data-testid="stHeader"] * { pointer-events: none !important; }
.block-container {
    max-width: 1500px;
    padding: 3rem 2.4rem 3rem 2.4rem;
}

/* No sidebar, no Streamlit menu / Deploy button */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"],
#MainMenu, [data-testid="stMainMenu"], [data-testid="stToolbar"],
[data-testid="stDeployButton"], .stDeployButton {
    display: none !important;
}

/* ---------- Typography ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
.stApp label, .stApp [data-testid="stMarkdownContainer"] {
    color: var(--text);
}
.stApp h2 { font-size: 1.9rem !important; font-weight: 800; letter-spacing: -.03em; margin-bottom: .8rem; }
.stApp h3 { font-size: 1.32rem !important; font-weight: 780; letter-spacing: -.02em; margin: 1.1rem 0 .6rem 0; }
.stApp h4 { font-size: 1.1rem !important; font-weight: 750; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p { color: var(--muted); }

/* ---------- Brand bar ---------- */
.brandbar { display: flex; align-items: center; gap: 1.1rem; flex-wrap: wrap; }
.brand-left { display: flex; align-items: center; gap: .85rem; }
.brand-mark {
    width: 44px; height: 44px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    color: var(--on-accent); font-size: 1.2rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 8px 22px var(--shadow);
}
.brand-name { color: var(--text); font-size: 1.22rem; font-weight: 850; letter-spacing: -.02em; line-height: 1.2; }
.brand-sub { color: var(--muted); font-size: .84rem; margin-top: .25rem; }
.status-chip {
    border: 1px solid var(--border); background: var(--surface);
    border-radius: 999px; color: var(--muted);
    padding: .42rem .85rem; font-size: .76rem; font-weight: 650; white-space: nowrap;
}

/* ---------- Pill radios (navigation, theme, scenario) ---------- */
div[role="radiogroup"] { gap: .45rem; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 999px; padding: .45rem 1.05rem; min-height: 40px; cursor: pointer;
    transition: border-color .15s ease, transform .15s ease;
}
div[role="radiogroup"] label:hover { border-color: var(--accent); transform: translateY(-1px); }
div[role="radiogroup"] label > *:not(:has(p)) { display: none !important; }
div[role="radiogroup"] label > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none !important; }
div[role="radiogroup"] label p { color: var(--text); font-weight: 650; font-size: .86rem; margin: 0; }
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border-color: transparent; box-shadow: 0 6px 18px var(--shadow);
}
.stApp div[role="radiogroup"] label:has(input:checked) p { color: var(--on-accent) !important; }

/* ---------- Cards ---------- */
.st-key-filters_bar, .st-key-roi_controls {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 8px 26px var(--shadow);
}
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 17px; padding: 1.05rem 1.15rem;
    box-shadow: 0 8px 26px var(--shadow); min-height: 118px;
    transition: transform .18s ease, border-color .18s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: var(--accent-2); }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    color: var(--muted) !important; font-size: .79rem; font-weight: 650;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important; font-size: 1.85rem; font-weight: 850;
}
[data-testid="stPlotlyChart"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 15px; overflow: hidden;
}
[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text); }

/* ---------- Key finding + insight lines ---------- */
.hero {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border-radius: 18px; padding: 1.25rem 1.5rem; margin: .4rem 0 1.1rem 0;
    box-shadow: 0 10px 28px var(--shadow);
}
.hero, .hero * { color: var(--on-accent) !important; }
.hero-label { font-size: .72rem; letter-spacing: .12em; font-weight: 800; opacity: .85; margin-bottom: .4rem; }
.hero-text { font-size: 1.18rem; line-height: 1.55; font-weight: 600; }
.insight {
    display: flex; gap: .75rem; align-items: flex-start;
    background: var(--surface-alt); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 12px;
    padding: .7rem 1rem; margin: .55rem 0 1.1rem 0;
    color: var(--text); font-size: .92rem; line-height: 1.5;
}
.insight-tag {
    background: var(--accent); color: var(--on-accent);
    font-size: .66rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
    padding: .2rem .55rem; border-radius: 6px; white-space: nowrap; margin-top: .12rem;
}

/* ---------- Tables ---------- */
.table-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; overflow: auto; margin-bottom: .4rem;
}
.data-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .86rem; color: var(--text); }
.data-table th {
    position: sticky; top: 0; z-index: 1;
    background: var(--surface-alt); color: var(--muted);
    text-align: left; font-weight: 700; font-size: .75rem; letter-spacing: .02em;
    padding: .7rem .9rem; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.data-table td { padding: .6rem .9rem; border-bottom: 1px solid var(--border); color: var(--text); }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover td { background: var(--surface-alt); }
.data-table th.num, .data-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
/* Streamlit's default table styling adds bright column lines; keep only soft row lines. */
.stApp .data-table, .stApp .data-table th, .stApp .data-table td {
    border-left: none !important; border-right: none !important; border-top: none !important;
}
.stApp .data-table th, .stApp .data-table td { border-bottom: 1px solid var(--border) !important; }
.stApp .data-table tbody tr:last-child td { border-bottom: none !important; }

/* Sliders: Streamlit paints them red; shift the hue to the dashboard's purple. */
[data-testid="stSlider"] [role="group"] { filter: hue-rotate(-98deg) saturate(1.15); }
.badge { display: inline-block; padding: .16rem .7rem; border-radius: 999px; font-weight: 750; font-size: .74rem; }
.badge-high { color: var(--high); background: color-mix(in srgb, var(--high) 16%, transparent); }
.badge-medium { color: var(--med); background: color-mix(in srgb, var(--med) 18%, transparent); }
.badge-low { color: var(--low); background: color-mix(in srgb, var(--low) 16%, transparent); }

/* ---------- Inputs, dropdowns, sliders ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="textarea"], [data-baseweb="select"] [role="combobox"] {
    background-color: var(--input-bg) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px;
}
[data-baseweb="select"] *, [data-baseweb="input"] *, [data-baseweb="textarea"] *,
[data-baseweb="select"] input {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}
[data-baseweb="select"] svg { fill: var(--muted) !important; }
/* Newer Streamlit versions (react-aria dropdowns) */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"] {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--text);
}
[data-testid="stSelectbox"] input::placeholder, [data-testid="stMultiSelect"] input::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}
[data-testid="stSelectbox"] button, [data-testid="stMultiSelect"] button,
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg { color: var(--muted) !important; }
[data-testid="stMultiSelectTagsContainer"] [data-tag] {
    background-color: var(--accent) !important; border-radius: 8px;
}
[data-testid="stMultiSelectTagsContainer"] [data-tag],
[data-testid="stMultiSelectTagsContainer"] [data-tag] * {
    color: var(--on-accent) !important; -webkit-text-fill-color: var(--on-accent) !important;
}
[data-baseweb="select"] [data-baseweb="tag"] {
    background-color: var(--accent) !important; border-radius: 8px;
}
[data-baseweb="select"] [data-baseweb="tag"], [data-baseweb="select"] [data-baseweb="tag"] * {
    color: var(--on-accent) !important;
    -webkit-text-fill-color: var(--on-accent) !important;
}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [role="listbox"] {
    background-color: var(--surface) !important; color: var(--text) !important;
    border-color: var(--border) !important;
}
[role="option"], [role="option"] * {
    background-color: var(--surface) !important; color: var(--text) !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
    background-color: var(--surface-alt) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* ---------- Alerts and buttons ---------- */
div[data-testid="stAlert"], div[data-testid="stAlert"] > div {
    background: var(--surface-alt) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
}
div[data-testid="stAlert"] *, div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}
.stButton button, .stDownloadButton button {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px;
    font-weight: 700; min-height: 42px;
}
.stButton button p, .stDownloadButton button p { color: var(--text) !important; }
.stButton button:hover, .stDownloadButton button:hover { border-color: var(--accent) !important; }

@media (max-width: 900px) {
    .block-container { padding: 1rem 1rem 2rem 1rem; }
}
"""

st.markdown(f"<style>{ROOT_VARS}{BASE_CSS}</style>", unsafe_allow_html=True)

# The chip only says "connected" after the database query has succeeded.
if db_connected:
    chip_text = f"PostgreSQL connected &nbsp;·&nbsp; {total_customers:,} records"
else:
    chip_text = "PostgreSQL not connected"

with header_left:
    st.markdown(
        f'<div class="brandbar"><div class="brand-left">'
        f'<div class="brand-mark">CI</div><div>'
        f'<div class="brand-name">Customer Intelligence</div>'
        f'<div class="brand-sub">Churn analytics &nbsp;·&nbsp; Customer value &nbsp;·&nbsp; Retention strategy</div>'
        f"</div></div>"
        f'<div class="status-chip">● &nbsp;{chip_text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

# -------------------- Chart renderer --------------------
def polish_chart(fig, height=390, legend=False):
    """Apply theme-aware chart styling and prevent wheel/drag zoom."""
    chart_text = "#52617A" if theme == "Light" else "#C2CEE0"
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=chart_text),
        margin=dict(l=18, r=22, t=28, b=42),
        showlegend=legend,
        hoverlabel=dict(bgcolor="#17233B", font=dict(color="#FFFFFF", size=12)),
    )
    if legend:
        fig.update_layout(
            legend=dict(orientation="h", y=1.1, x=0, title=None, font=dict(color=chart_text))
        )
    fig.update_xaxes(
        fixedrange=True, showline=False, gridcolor="rgba(135,153,180,.20)",
        zeroline=False, automargin=True,
        tickfont=dict(color=chart_text), title_font=dict(color=chart_text),
    )
    fig.update_yaxes(
        fixedrange=True, showline=False, gridcolor="rgba(135,153,180,.20)",
        zeroline=False, automargin=True,
        tickfont=dict(color=chart_text), title_font=dict(color=chart_text),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False},
    )

# -------------------- Page navigation --------------------
pages = [
    "Overview",
    "Why customers churn",
    "Survival analysis",
    "Take action",
]
page = st.radio(
    "Workspace",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="dashboard_page_v2",
)

# -------------------- Filters (apply to KPIs, charts and the simulator) --------------------
TENURE_BANDS = {
    "All tenures": None,
    "0–12 months": (0, 12),
    "13–24 months": (13, 24),
    "25–48 months": (25, 48),
    "49+ months": (49, 10_000),
}

with st.container(key="filters_bar"):
    f1, f2, f3 = st.columns([2.4, 1.6, 1.4], vertical_alignment="center")
    with f1:
        # Leave empty to include every contract type.
        selected_contracts = st.multiselect(
            "Contract type",
            contract_names,
            default=[],
            placeholder="All contract types",
        ) if contract_col else []
    with f2:
        tenure_choice = st.selectbox(
            "Customer tenure", list(TENURE_BANDS.keys())
        ) if tenure_col else "All tenures"

view_df = customers_df
if contract_col and selected_contracts:
    view_df = view_df[view_df[contract_col].astype(str).isin(selected_contracts)]
if tenure_col and TENURE_BANDS[tenure_choice] is not None:
    low, high = TENURE_BANDS[tenure_choice]
    view_df = view_df[view_df[tenure_col].between(low, high)]

with f3:
    st.caption(f"Showing **{len(view_df):,}** of {total_customers:,} customers")

if view_df.empty:
    st.warning("No customers match these filters. Widen the filters above.")
    st.stop()

filters_active = len(view_df) != total_customers
view_total = len(view_df)
view_churned = int(view_df["_churned"].sum())
view_churn_rate = view_churned / view_total

# -------------------- Shared calculations --------------------
def high_risk_revenue(risk_info, df):
    """Monthly revenue and customer count of active customers flagged High risk."""
    if risk_info is None or not customer_id_col or not monthly_col:
        return None
    risk_df_ = risk_info["df"]
    high_ids = set(
        risk_df_.loc[
            risk_df_[risk_info["level"]].astype(str).str.strip().str.title() == "High",
            risk_info["id"],
        ].astype(str)
    )
    if not customers_df[customer_id_col].astype(str).isin(high_ids).any():
        return None
    active = df[~df["_churned"]]
    match = active[active[customer_id_col].astype(str).isin(high_ids)]
    return float(match[monthly_col].sum()), int(len(match))

def build_headline(df):
    parts = []
    rate = float(df["_churned"].mean())

    rates = None
    if contract_col:
        rates = df.dropna(subset=[contract_col]).groupby(contract_col)["_churned"].mean()

    if rates is not None and len(rates) >= 2 and rates.max() > rates.min():
        hi, lo = rates.idxmax(), rates.idxmin()
        hi_rate, lo_rate = f"{rates[hi]:.1%}", f"{rates[lo]:.1%}"
        parts.append(
            f"{b(hi)} customers churn at {b(hi_rate)}, compared with "
            f"{b(lo_rate)} on {b(lo)} contracts."
        )
    else:
        parts.append(f"{b(f'{rate:.1%}')} of customers in this view have churned.")

    if monthly_col:
        lost = float(df.loc[df["_churned"], monthly_col].sum())
        parts.append(
            f"Churn has already cost {b(money0(lost))} per month in recurring revenue."
        )

    at_risk = high_risk_revenue(risk, df)
    if at_risk is not None and at_risk[1] > 0:
        count_text = f"{at_risk[1]:,}"
        parts.append(
            f"A further {b(money0(at_risk[0]))} per month sits with "
            f"{b(count_text)} high-risk customers who are still active."
        )
    return " ".join(parts)

# -------------------- Overview --------------------
if page == "Overview":
    st.subheader("Business overview")

    st.markdown(
        safe(
            f'<div class="hero"><div class="hero-label">KEY FINDING</div>'
            f'<div class="hero-text">{build_headline(view_df)}</div></div>'
        ),
        unsafe_allow_html=True,
    )

    lost_revenue = (
        float(view_df.loc[view_df["_churned"], monthly_col].sum()) if monthly_col else None
    )
    at_risk = high_risk_revenue(risk, view_df)

    a, b_col, c, d = st.columns(4)
    a.metric("Customers in view", f"{view_total:,}")
    b_col.metric(
        "Churn rate",
        f"{view_churn_rate:.1%}",
        delta=(
            f"{(view_churn_rate - churn_rate) * 100:+.1f} pts vs all customers"
            if filters_active else None
        ),
        delta_color="inverse",
    )
    c.metric(
        "Monthly revenue lost to churn",
        money0(lost_revenue) if lost_revenue is not None else "n/a",
        help="Monthly charges of customers who have already churned.",
    )
    d.metric(
        "Revenue at risk",
        money0(at_risk[0]) if at_risk is not None else "n/a",
        help=(
            "Monthly charges of active customers flagged High risk by the model."
            if at_risk is not None
            else "Add customer_churn_risk.csv to the outputs/ folder to see this."
        ),
    )

    st.caption("Metrics are calculated from customer records in PostgreSQL.")

    left, right = st.columns(2)

    with left:
        st.markdown("### Churn status")
        churn_plot = pd.DataFrame(
            {
                "Status": ["Stayed", "Churned"],
                "Customers": [view_total - view_churned, view_churned],
            }
        )
        fig = px.bar(
            churn_plot, x="Status", y="Customers", color="Status",
            color_discrete_map={"Stayed": STAY_COLOR, "Churned": CHURN_COLOR},
            text="Customers",
        )
        fig.update_traces(marker_line_width=0, texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_yaxes(range=[0, float(churn_plot["Customers"].max()) * 1.18], title=None)
        fig.update_xaxes(title=None)
        polish_chart(fig, 380)
        insight(
            f"{b(f'{view_churn_rate:.1%}')} of customers in this view have churned "
            f"({view_churned:,} of {view_total:,})."
        )

    with right:
        st.markdown("### Contract mix")
        if contract_col:
            contract_counts = view_df[contract_col].fillna("Unknown").astype(str).value_counts()
            contract_plot = contract_counts.rename_axis("Contract").reset_index()
            contract_plot.columns = ["Contract", "Customers"]
            fig = px.bar(
                contract_plot, x="Contract", y="Customers",
                color="Contract", color_discrete_map=contract_color_map,
                text="Customers",
            )
            fig.update_traces(marker_line_width=0, texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_yaxes(range=[0, float(contract_plot["Customers"].max()) * 1.18], title=None)
            fig.update_xaxes(title=None)
            polish_chart(fig, 380)
            top_name = contract_counts.idxmax()
            top_share = contract_counts.max() / contract_counts.sum()
            insight(
                f"{b(top_name)} is the largest group, covering "
                f"{b(f'{top_share:.0%}')} of customers in this view."
            )

    st.markdown("### Customer value summary")
    if clv_df is not None:
        render_table(clv_df, height=320)
    else:
        st.info("Add clv_summary.csv to the outputs/ folder.")

# -------------------- Why customers churn --------------------
elif page == "Why customers churn":
    st.subheader("Why customers churn")

    a, b_col = st.columns(2)
    a.metric("Churned customers", f"{view_churned:,}")
    b_col.metric("Observed churn rate", f"{view_churn_rate:.1%}")

    st.markdown("### Churn by contract")
    if contract_col:
        contract_churn = (
            view_df.groupby(contract_col, dropna=False)["_churned"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        contract_plot = contract_churn.rename("Churn rate").reset_index()
        contract_plot.columns = ["Contract", "Churn rate"]
        contract_plot["Contract"] = contract_plot["Contract"].astype(str)
        contract_plot = contract_plot.sort_values("Churn rate")
        fig = px.bar(
            contract_plot, x="Churn rate", y="Contract", orientation="h",
            color_discrete_sequence=[CHURN_COLOR],
            text=contract_plot["Churn rate"].map(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(marker_line_width=0, textposition="outside")
        fig.update_xaxes(
            title=None, ticksuffix="%",
            range=[0, max(float(contract_plot["Churn rate"].max()), 1) * 1.22],
        )
        fig.update_yaxes(title=None)
        polish_chart(fig, 340)

        if len(contract_plot) >= 2:
            hi_row = contract_plot.iloc[-1]
            lo_row = contract_plot.iloc[0]
            if lo_row["Churn rate"] > 0:
                ratio_text = f"{hi_row['Churn rate'] / lo_row['Churn rate']:.1f}×"
                insight(
                    f"{b(hi_row['Contract'])} customers churn at {hi_row['Churn rate']:.1f}%, "
                    f"{b(ratio_text)} the rate of {b(lo_row['Contract'])} "
                    f"({lo_row['Churn rate']:.1f}%). Closing this gap is the clearest place "
                    f"to focus retention effort."
                )
            else:
                insight(
                    f"{b(hi_row['Contract'])} customers churn at {hi_row['Churn rate']:.1f}%, "
                    f"while {b(lo_row['Contract'])} customers show no churn in this view."
                )
        else:
            insight("Only one contract group is selected, so there is nothing to compare.")

    st.markdown("### What drives churn (SHAP)")
    st.caption(
        "SHAP importance shows how much the model relies on each feature when it predicts churn. "
        "It describes the model's behaviour, not cause and effect: a high-ranked feature is "
        "linked to churn in the data, but changing it may not stop customers leaving."
    )
    if shap_df is not None:
        numeric_cols = shap_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            feature_col = next(
                (col for col in shap_df.columns if "feature" in col.lower()),
                shap_df.columns[0],
            )
            value_col = numeric_cols[0]
            plot_df = (
                shap_df[[feature_col, value_col]]
                .dropna()
                .set_index(feature_col)
            )
            shap_plot = plot_df.head(15).reset_index()
            shap_plot.columns = ["Feature", "Importance"]
            shap_plot = shap_plot.sort_values("Importance", ascending=True)
            fig = px.bar(
                shap_plot, x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            polish_chart(fig, 440)
            if len(shap_plot) >= 2:
                insight(
                    f"The model relies most on {b(shap_plot.iloc[-1]['Feature'])}, "
                    f"followed by {b(shap_plot.iloc[-2]['Feature'])}."
                )
            st.caption(
                "SHAP shows how much each feature influences the model's predictions. "
                "It does not prove that a feature causes churn, so use it to prioritise "
                "what to investigate, not as proof of cause."
            )
        with st.expander("Show SHAP table"):
            render_table(shap_df, height=360)

    st.markdown("### How accurate is the model?")
    metrics_df = load_csv("model_evaluation.csv")
    if metrics_df is not None:
        render_table(metrics_df, height=260)
    else:
        st.info("Add model_evaluation.csv to the outputs/ folder with actual model metrics.")

    confusion_df = load_csv("confusion_matrix.csv")
    if confusion_df is not None:
        st.markdown("#### Confusion matrix")
        render_table(confusion_df, height=260)
    else:
        st.info("Add confusion_matrix.csv to the outputs/ folder with actual confusion matrix values.")

    st.markdown("### Who your customers are")
    if cluster_df is not None:
        st.markdown("#### Customer cluster profiles")
        render_table(cluster_df, height=320)

    if contract_df is not None:
        st.markdown("#### Contract retention summary")
        render_table(contract_df, height=320)

    st.markdown("### Browse customer records")
    if customer_id_col:
        customer_options = view_df[customer_id_col].astype(str).tolist()
        selected_customer = st.selectbox("Select a customer", customer_options)

        customer_row = view_df[
            view_df[customer_id_col].astype(str) == selected_customer
        ].iloc[0]

        record = customer_row.drop(labels=["_churned"])
        record_df = pd.DataFrame(
            {
                "Field": [pretty_label(name) for name in record.index],
                "Value": [str(value) for value in record.values],
            }
        )
        render_table(record_df, max_rows=100, height=420)
    else:
        render_table(view_df.drop(columns=["_churned"]).head(100), height=420)

# -------------------- Survival analysis --------------------
elif page == "Survival analysis":
    st.subheader("Survival analysis")

    st.markdown("### How long do customers stay? (Kaplan–Meier)")

    if tenure_col is None:
        st.warning("The customer table has no tenure column.")
    else:
        surv_cols = [tenure_col, churn_col] + ([contract_col] if contract_col else [])
        surv = None
        try:
            surv = run_survival(view_df[surv_cols], tenure_col, churn_col, contract_col)
        except ImportError:
            st.warning(
                "Kaplan–Meier and Cox analysis require lifelines. "
                "Install it with: pip install lifelines"
            )
        except Exception as error:
            st.warning(f"Kaplan–Meier analysis could not be calculated: {error}")

        if surv is not None:
            fig = px.line(
                surv["curve"], x="Tenure (months)", y="Churn-free probability",
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig.update_traces(line=dict(width=3), hovertemplate="Month %{x}<br>Churn-free: %{y:.1%}<extra></extra>")
            fig.update_yaxes(title=None, tickformat=".0%", range=[0, 1.02])
            fig.update_xaxes(title="Tenure (months)")
            polish_chart(fig, 400)

            m1, m2 = st.columns(2)
            m1.metric("12-month churn-free estimate", f"{surv['s12'] * 100:.2f}%")
            m2.metric("24-month churn-free estimate", f"{surv['s24'] * 100:.2f}%")

            s12_text = f"{surv['s12']:.0%}"
            s24_text = f"{surv['s24']:.0%}"
            insight(
                f"About {b(s12_text)} of customers are still active after 12 months "
                f"and {b(s24_text)} after 24 months."
            )
            st.caption(
                "Kaplan–Meier estimates the probability of remaining churn-free "
                "over tenure, accounting for customers who have not churned yet."
            )

            if surv["contract_long"] is not None and not surv["contract_long"].empty:
                st.markdown("#### Survival by contract type")
                fig = px.line(
                    surv["contract_long"], x="Tenure (months)", y="Churn-free probability",
                    color="Contract", color_discrete_map=contract_color_map,
                )
                fig.update_traces(line=dict(width=3), hovertemplate="%{fullData.name}<br>Month %{x}<br>Churn-free: %{y:.1%}<extra></extra>")
                fig.update_yaxes(title=None, tickformat=".0%", range=[0, 1.02])
                fig.update_xaxes(title="Tenure (months)")
                polish_chart(fig, 420, legend=True)

                s24 = surv["contract_s24"]
                if len(s24) >= 2:
                    worst = min(s24, key=s24.get)
                    best = max(s24, key=s24.get)
                    insight(
                        f"{b(worst)} customers have only a {b(f'{s24[worst]:.0%}')} chance of "
                        f"staying churn-free to month 24, versus {b(f'{s24[best]:.0%}')} "
                        f"for {b(best)}."
                    )

            if surv["p_value"] is not None:
                if surv["p_value"] < 0.0001:
                    st.write("Contract-group log-rank test p-value: < 0.0001")
                else:
                    st.write(f"Contract-group log-rank test p-value: {surv['p_value']:.4f}")
            elif surv["p_error"]:
                st.caption(f"Could not calculate the log-rank test: {surv['p_error']}")

    st.markdown("### Cox proportional-hazards model")

    if tenure_col is None:
        st.warning("The customer table has no tenure column.")
    else:
        cox_cols = [
            col for col in dict.fromkeys([tenure_col, churn_col] + COX_CANDIDATES)
            if col in view_df.columns
        ]
        try:
            cox = run_cox(view_df[cox_cols], tenure_col, churn_col)
            if cox["warning"]:
                st.warning(cox["warning"])
            else:
                st.caption(
                    "Hazard ratio above 1 indicates a higher estimated churn "
                    "hazard; below 1 indicates a lower estimated hazard. "
                    "These are associations, not causal effects."
                )
                cox_results = cox["results"]
                display_cols = [
                    col for col in [
                        "Feature", "Hazard ratio", "HR 95% lower",
                        "HR 95% upper", "p-value",
                    ]
                    if col in cox_results.columns
                ]
                render_table(cox_results[display_cols], decimals=4, height=360)

                if cox.get("ph_flagged"):
                    st.warning(
                        "Proportional-hazards check: the effect of "
                        + ", ".join(cox["ph_flagged"])
                        + " appears to change over tenure (p < 0.05). "
                        "Read these hazard ratios as averages over time, not fixed effects."
                    )
                elif cox.get("ph_error"):
                    st.caption(
                        "The proportional-hazards check could not be run: "
                        f"{cox['ph_error']}"
                    )
                else:
                    st.caption(
                        "Proportional-hazards check passed for every feature (p ≥ 0.05)."
                    )

                if {"Hazard ratio", "p-value"}.issubset(cox_results.columns):
                    sig = cox_results[cox_results["p-value"] < 0.05]
                    if sig.empty:
                        insight("No feature has a statistically significant link to churn hazard at the 5% level.")
                    else:
                        top = sig.loc[sig["Hazard ratio"].idxmax()]
                        bottom = sig.loc[sig["Hazard ratio"].idxmin()]
                        if top["Hazard ratio"] > 1 and bottom["Hazard ratio"] < 1:
                            insight(
                                f"{b(top['Feature'])} is linked to the highest churn hazard "
                                f"(HR {top['Hazard ratio']:.2f}); {b(bottom['Feature'])} is the most "
                                f"protective (HR {bottom['Hazard ratio']:.2f})."
                            )
                        elif top["Hazard ratio"] > 1:
                            insight(
                                f"{b(top['Feature'])} is linked to the highest churn hazard "
                                f"(HR {top['Hazard ratio']:.2f})."
                            )
                        else:
                            insight(
                                f"{b(bottom['Feature'])} is the most protective feature "
                                f"(HR {bottom['Hazard ratio']:.2f})."
                            )

                st.download_button(
                    "Download Cox model results",
                    cox_results.to_csv(index=False),
                    "cox_model_results.csv",
                    "text/csv",
                )
        except ImportError:
            st.warning(
                "Cox model analysis requires lifelines. "
                "Install it with: pip install lifelines"
            )
        except Exception as error:
            st.warning(
                "The Cox model could not be fitted for this selection. This usually means "
                "too few customers or churn events in the current filters, or features that "
                "are almost perfectly correlated. Try widening the filters. "
                f"Details: {error}"
            )

# -------------------- Take action --------------------
elif page == "Take action":
    st.subheader("Take action")

    # ---------- Customer-level churn risk ----------
    st.markdown("### Who to contact first")

    if risk_error:
        st.error(
            "customer_churn_risk.csv must contain customerID, "
            "churn_probability, risk_level, and contact_focus."
        )
    elif risk is None:
        st.warning(
            "customer_churn_risk.csv was not found, so the customer risk list is not available. "
            "The Revenue at risk figure on the Overview page will also show n/a. "
            "Export the file from Colab and place it in the outputs/ folder to turn these on."
        )
    else:
        risk_df = risk["df"]
        id_col, probability_col = risk["id"], risk["prob"]
        level_col, focus_col = risk["level"], risk["focus"]

        # ---- ID check: risk CSV vs PostgreSQL ----
        match = risk.get("match")
        if match is None:
            st.caption("Customer IDs could not be checked against PostgreSQL (no customer ID column).")
        elif match["risk_matched"] == 0:
            st.error(
                "None of the customer IDs in customer_churn_risk.csv match the PostgreSQL customers. "
                "The risk file may be from a different dataset, or the ID column is wrong. "
                "Revenue at risk cannot be calculated until this is fixed."
            )
        elif match["risk_matched"] < match["risk_total"]:
            missing = match["risk_total"] - match["risk_matched"]
            st.warning(
                f"{missing:,} of {match['risk_total']:,} customer IDs in the risk file were not found in "
                f"PostgreSQL. Those customers are left out of Revenue at risk and the filters."
            )
        elif match["db_covered"] < match["db_total"]:
            st.caption(
                f"Customer IDs check: all {match['risk_total']:,} risk-file IDs match PostgreSQL. "
                f"The file scores {match['db_covered']:,} of {match['db_total']:,} customers."
            )
        else:
            st.caption(
                f"Customer IDs check passed: all {match['risk_total']:,} customers match PostgreSQL."
            )

        risk_view = risk_df
        if filters_active and customer_id_col:
            view_ids = set(view_df[customer_id_col].astype(str))
            risk_view = risk_df[risk_df[id_col].astype(str).isin(view_ids)]

        st.caption(
            "Sorted by model-estimated churn probability. "
            "These are risk estimates, not guaranteed churn outcomes "
            "or proof that a retention action will work."
        )

        sel_col, _ = st.columns([1, 3])
        with sel_col:
            selected_risk = st.selectbox(
                "Filter by risk level",
                ["All", "High", "Medium", "Low"],
            )

        filtered_risk_df = (
            risk_view
            if selected_risk == "All"
            else risk_view[
                risk_view[level_col].astype(str).str.title() == selected_risk
            ]
        )

        if filtered_risk_df.empty:
            st.info("No scored customers match these filters.")
        else:
            shown = min(300, len(filtered_risk_df))
            render_table(
                filtered_risk_df[[id_col, probability_col, level_col, focus_col]],
                max_rows=300,
                height=390,
                badge_col=level_col,
                rename={
                    id_col: "Customer ID",
                    probability_col: "Churn probability",
                    level_col: "Risk level",
                    focus_col: "Contact focus",
                },
            )
            st.caption(
                f"Showing the top {shown:,} of {len(filtered_risk_df):,} customers. "
                "Download the file for the full list."
            )

            at_risk = high_risk_revenue(risk, view_df)
            if at_risk is not None and at_risk[1] > 0:
                insight(
                    f"{b(f'{at_risk[1]:,}')} active customers in this view are flagged High risk, "
                    f"holding {b(money0(at_risk[0]))} of monthly revenue. "
                    f"Start outreach at the top of this list."
                )

        st.download_button(
            "Download customer churn risk",
            risk_df.to_csv(index=False),
            "customer_churn_risk.csv",
            "text/csv",
        )

    # ---------- ROI Simulator ----------
    st.markdown("### Retention ROI simulator")
    st.caption(
        "Pick a scenario, then fine-tune it. This is an illustrative what-if, not a measured result. "
        "The discount is given to every contacted customer. See the assumptions below."
    )

    ROI_PRESETS = {
        "Cautious": {"target_pct": 25, "discount": 10, "months": 3, "saved": 3, "extra": 4},
        "Balanced": {"target_pct": 25, "discount": 10, "months": 3, "saved": 5, "extra": 6},
        "Optimistic": {"target_pct": 40, "discount": 10, "months": 3, "saved": 10, "extra": 12},
    }

    def apply_roi_preset():
        values = ROI_PRESETS[st.session_state["roi_preset"]]
        for name, value in values.items():
            st.session_state[f"roi_{name}"] = value

    if "roi_preset" not in st.session_state:
        st.session_state["roi_preset"] = "Balanced"
        apply_roi_preset()

    st.radio(
        "Start from a scenario",
        list(ROI_PRESETS.keys()),
        horizontal=True,
        key="roi_preset",
        on_change=apply_roi_preset,
    )

    segment_count = len(view_df)
    segment_avg_charge = (
        float(view_df[monthly_col].mean())
        if monthly_col and not view_df.empty
        else avg_monthly_charge
    )

    ctrl_col, result_col = st.columns([1, 1.15], gap="large")

    with ctrl_col:
        with st.container(key="roi_controls"):
            st.markdown(f"**Customer group:** {segment_count:,} customers")
            st.caption("Change the group with the filters at the top of the page.")
            target_pct = st.slider(
                "Customers contacted (% of group)", 5, 100, key="roi_target_pct",
                help="Share of the selected customers who receive the offer.",
            )
            discount_pct = st.slider(
                "Discount offered (%)", 0, 50, key="roi_discount",
                help="Percentage taken off the monthly charge.",
            )
            discount_months = st.slider(
                "Discount lasts (months)", 1, 12, key="roi_months",
            )
            churn_reduction_pct = st.slider(
                "Contacted customers saved (%)", 1, 20, key="roi_saved",
                help="Of the customers contacted, the share who stay because of the offer but would otherwise have left.",
            )
            extra_months = st.slider(
                "Extra months they stay", 1, 24, key="roi_extra",
            )

    target_count = max(1, round(segment_count * target_pct / 100))
    additional_retained = target_count * churn_reduction_pct / 100
    revenue_preserved = additional_retained * segment_avg_charge * extra_months
    discount_cost = (
        target_count
        * segment_avg_charge
        * discount_pct / 100
        * discount_months
    )

    net_benefit = revenue_preserved - discount_cost
    roi = net_benefit / discount_cost * 100 if discount_cost else 0

    with result_col:
        r1, r2 = st.columns(2)
        r1.metric("Revenue preserved", money(revenue_preserved))
        r2.metric("Discount cost", money(discount_cost))
        r3, r4 = st.columns(2)
        r3.metric(
            "Net revenue benefit", money(net_benefit),
            help="Revenue preserved minus discount cost. This is revenue, not profit.",
        )
        r4.metric(
            "Revenue-based ROI", f"{roi:.1f}%",
            help="(Revenue preserved − discount cost) ÷ discount cost. Based on revenue, not profit: "
                 "margins and the cost of running the campaign are not included.",
        )

        outcome = "gain" if net_benefit >= 0 else "loss"
        insight(
            f"Contact {b(f'{target_count:,}')} customers with {b(f'{discount_pct}% off')} for "
            f"{b(f'{discount_months} months')}. If {b(f'{churn_reduction_pct}%')} of them stay "
            f"{b(f'{extra_months} extra months')}, you keep {b(money0(revenue_preserved))} "
            f"against {b(money0(discount_cost))} in discounts, a net {outcome} of "
            f"{b(money0(abs(net_benefit)))}."
        )

    left, right = st.columns(2)

    with left:
        st.markdown("### Savings vs. campaign cost")
        roi_plot = pd.DataFrame({
            "Category": ["Revenue preserved", "Discount cost"],
            "Amount ($)": [revenue_preserved, discount_cost],
        })
        fig = px.bar(
            roi_plot, x="Amount ($)", y="Category", orientation="h",
            color="Category",
            color_discrete_map={"Revenue preserved": STAY_COLOR, "Discount cost": GOLD_COLOR},
            text="Amount ($)",
        )
        fig.update_traces(marker_line_width=0, texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_xaxes(
            title=None, tickprefix="$",
            range=[0, max(revenue_preserved, discount_cost, 1) * 1.25],
        )
        fig.update_yaxes(title=None)
        polish_chart(fig, 340)

        break_even = discount_pct * discount_months / extra_months
        if discount_pct == 0:
            insight("With no discount, the campaign has no cost to recover.")
        else:
            insight(
                f"At this discount, the campaign breaks even if about "
                f"{b(f'{break_even:.1f}%')} of contacted customers are saved."
            )

    with right:
        st.markdown("### Scenario details")
        st.metric(
            "Estimated additional retained",
            f"{additional_retained:,.1f}",
            help="Expected value: customers contacted × share saved. It can be fractional and is used unrounded in the money calculations.",
        )
        st.metric("Group average monthly charge", money(segment_avg_charge))

        if net_benefit > 0:
            st.success("Estimated revenue preserved exceeds discount cost.")
        elif net_benefit < 0:
            st.warning("Estimated discount cost exceeds revenue preserved.")
        else:
            st.info("Estimated revenue preserved equals discount cost.")

    st.info(
        "**Illustrative what-if, not a result.** ROI here is revenue-based: "
        "(revenue preserved − discount cost) ÷ discount cost. It is not profit, because margins and "
        "campaign running costs are not included. Do not quote these numbers as measured savings."
    )

    with st.expander("Assumptions behind this calculator (read before quoting the numbers)"):
        st.markdown(
            "- **Illustrative only.** This is a what-if scenario, not a measured campaign result or a forecast.\n"
            "- **Who gets the offer:** the contacted customers are treated as a random share of the selected group. "
            "In real life you would target by churn risk, which usually works better.\n"
            "- **Everyone contacted gets the discount**, including customers who would have stayed anyway. "
            "That is why the discount cost is calculated on all contacted customers.\n"
            "- **Customers saved** (the slider) is an assumption. It is the share of contacted customers who stay "
            "because of the offer but would otherwise have left.\n"
            "- **Revenue preserved** = customers saved × average monthly charge × extra months they stay. "
            "Customer counts are used unrounded, so they can be fractional.\n"
            "- **Not included:** cost of running the campaign, customers who leave anyway after the extra months, "
            "and any change in future behaviour.\n"
            "- The customer group and average charge come from PostgreSQL."
        )

    # ---------- Saved campaign scenario ----------
    st.markdown("### Saved campaign scenario")

    if roi_scenario_df is not None:
        render_table(roi_scenario_df, height=300)
    else:
        st.info("Add retention_roi_scenario.csv to the outputs/ folder.")

    sensitivity_path = APP_DIR / "outputs" / "retention_roi_sensitivity.png"
    if not sensitivity_path.exists():
        sensitivity_path = APP_DIR / "retention_roi_sensitivity.png"
    if sensitivity_path.exists():
        st.markdown("### Saved sensitivity analysis")
        st.image(
            str(sensitivity_path),
            caption="Sensitivity analysis exported from the project notebook.",
            use_container_width=True,
        )