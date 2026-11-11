import os
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
)

APP_DIR = Path(__file__).resolve().parent

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

def load_csv(filename):
    possible_paths = [
        APP_DIR / "csv files" / filename,
        APP_DIR / filename
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as error:
                st.warning(f"Could not read {filename}: {error}")
                return None

    st.warning(f"CSV not found: {filename}")
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

def money(value):
    return f"${value:,.2f}"


# -------------------- Premium chart renderer --------------------
CHART_COLORS = ["#6546D7", "#168EAD", "#F08A5D", "#3A9D8F", "#D5A441"]

def polish_chart(fig, height=390):
    """Apply consistent dashboard styling and prevent wheel/drag zoom."""
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#66758F"),
        margin=dict(l=18, r=22, t=28, b=42),
        showlegend=False,
        hoverlabel=dict(bgcolor="#17233B", font=dict(color="#FFFFFF", size=12)),
    )
    fig.update_xaxes(
        fixedrange=True,
        showline=False,
        gridcolor="rgba(135,153,180,.16)",
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        fixedrange=True,
        showline=False,
        gridcolor="rgba(135,153,180,.16)",
        zeroline=False,
        automargin=True,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False},
    )

# -------------------- Load data --------------------
try:
    customers_df = load_customers()
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

customers_df["_churned"] = (
    customers_df[churn_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["yes", "true", "1", "churned"])
)

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

clv_df = load_csv("clv_summary.csv")

# -------------------- Premium interface and navigation --------------------
# Appearance controls presentation only. Database, models, calculations,
# analysis and page-level functionality below remain unchanged.
theme = st.sidebar.radio(
    "THEME",
    ["Light", "Dark"],
    horizontal=True,
    index=0,
    key="dashboard_theme",
)

if theme == "Dark":
    palette = {
        "app": "#090F1F",
        "surface": "#111B2E",
        "surface_alt": "#192943",
        "border": "#2B3B56",
        "text": "#F1F5FC",
        "muted": "#A7B5CA",
        "accent": "#A78BFA",
        "accent2": "#38BDF8",
        "shadow": "rgba(3,10,25,.34)",
        "input": "#182338",
    }
else:
    palette = {
        "app": "#F2F5FB",
        "surface": "#FFFFFF",
        "surface_alt": "#F7F8FF",
        "border": "#E3E8F2",
        "text": "#17233B",
        "muted": "#66758F",
        "accent": "#6546D7",
        "accent2": "#168EAD",
        "shadow": "rgba(31,48,78,.09)",
        "input": "#FFFFFF",
    }

st.markdown(
    f"""
    <style>
    :root {{
        --app-bg: {palette["app"]};
        --surface: {palette["surface"]};
        --surface-alt: {palette["surface_alt"]};
        --border: {palette["border"]};
        --text: {palette["text"]};
        --muted: {palette["muted"]};
        --accent: {palette["accent"]};
        --accent-2: {palette["accent2"]};
        --shadow: {palette["shadow"]};
        --input-bg: {palette["input"]};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background: var(--app-bg);
        color: var(--text);
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Hide Streamlit's top-right three-dot main menu */
    #MainMenu,
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] button,
    button[title="Main menu"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background: var(--app-bg);
    }}
    .block-container {{
        max-width: 1660px;
        padding: 1.35rem 2.6rem 3rem 2.6rem;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--surface);
        border-right: 1px solid var(--border);
        min-width: 265px;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem;
    }}
    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: var(--muted);
    }}

    /* Main brand header */
    .brandbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .3rem 0 1.25rem 0;
        margin-bottom: 1.1rem;
        border-bottom: 1px solid var(--border);
    }}
    .brand-left {{
        display: flex;
        align-items: center;
        gap: .85rem;
    }}
    .brand-mark {{
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 8px 24px var(--shadow);
        box-shadow: 0 8px 22px var(--shadow);
    }}
    .brand-name {{
        color: var(--text);
        font-size: 1.22rem;
        font-weight: 850;
        letter-spacing: -.02em;
        line-height: 1.2;
    }}
    .brand-sub {{
        color: var(--muted);
        font-size: .84rem;
        margin-top: .28rem;
    }}
    .status-chip {{
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 999px;
        color: var(--muted);
        padding: .48rem .85rem;
        font-size: .76rem;
        font-weight: 650;
    }}

    /* Navigation */
    div[role="radiogroup"] {{
        gap: .45rem;
        flex-wrap: wrap;
    }}
    div[role="radiogroup"] label {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: .52rem .85rem;
        min-height: 42px;
    }}
    div[role="radiogroup"] label p {{
        color: var(--text);
        font-weight: 650;
        font-size: .84rem;
    }}

    /* Typography */
    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {{
        color: var(--text);
    }}
    h1 {{
        letter-spacing: -.045em;
        font-weight: 850;
        font-size: 2.65rem !important;
        line-height: 1.12;
    }}
    h2 {{
        letter-spacing: -.03em;
        font-weight: 800;
        font-size: 2.05rem !important;
        line-height: 1.2;
        margin-top: .45rem;
        margin-bottom: 1rem;
    }}
    h3 {{
        letter-spacing: -.025em;
        font-weight: 780;
        font-size: 1.48rem !important;
        line-height: 1.3;
        margin-top: 1.15rem;
        margin-bottom: .7rem;
    }}
    h4 {{
        font-size: 1.18rem !important;
        font-weight: 750;
    }}
    [data-testid="stCaptionContainer"], .stCaption {{
        color: var(--muted);
    }}

    /* KPI cards */
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 8px 26px var(--shadow);
        min-height: 124px;
        transition: transform .18s ease, border-color .18s ease,
                    box-shadow .18s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        border-color: var(--accent-2);
        box-shadow: 0 14px 32px var(--shadow);
    }}
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        color: var(--muted) !important;
        font-size: .79rem;
        font-weight: 650;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--text) !important;
        font-size: 1.95rem;
        font-weight: 850;
    }}

    /* Content panels, charts and tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stPlotlyChart"],
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        overflow: hidden;
    }}
    [data-testid="stExpander"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
    }}
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] {{
        background: var(--input-bg);
        border-color: var(--border);
        color: var(--text);
        border-radius: 10px;
    }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid var(--border);
        font-weight: 700;
        min-height: 42px;
    }}
    hr {{
        border-color: var(--border);
    }}

    @media (max-width: 900px) {{
        .block-container {{
            padding: 1rem 1rem 2rem 1rem;
        }}
        .brandbar {{
            align-items: flex-start;
            flex-direction: column;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="brandbar">
        <div class="brand-left">
            <div class="brand-mark">CI</div>
            <div>
                <div class="brand-name">Customer Intelligence</div>
                <div class="brand-sub">Churn analytics&nbsp; · &nbsp;Customer value&nbsp; · &nbsp;Retention strategy</div>
            </div>
        </div>
        <div class="status-chip">● &nbsp; PostgreSQL connected &nbsp;·&nbsp; {total_customers:,} records</div>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    "Overview",
    "Churn Analysis",
    "Customer Insights",
    "Retention Campaigns",
    "Retention ROI Simulator",
]
page = st.radio(
    "Workspace",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="dashboard_page",
)

st.markdown("---")

# -------------------- Overview --------------------
if page == "Overview":
    st.subheader("Business overview")

    a, b, c, d = st.columns(4)
    a.metric("Total customers", f"{total_customers:,}")
    b.metric("Churn rate", f"{churn_rate:.1%}")
    c.metric("Monthly charges", money(monthly_revenue))
    d.metric("Average monthly charge", money(avg_monthly_charge))

    st.caption(
        "Metrics are calculated from customer records in PostgreSQL."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Churn status")
        churn_summary = pd.DataFrame(
            {
                "Customers": [
                    total_customers - churned_customers,
                    churned_customers,
                ]
            },
            index=["Stayed", "Churned"],
        )
        churn_plot = churn_summary.reset_index()
        churn_plot.columns = ["Status", "Customers"]
        fig = px.bar(
            churn_plot, x="Status", y="Customers", color="Status",
            color_discrete_map={"Stayed": "#168EAD", "Churned": "#F08A5D"},
            text="Customers",
        )
        fig.update_traces(marker_line_width=0, texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_yaxes(range=[0, float(churn_plot["Customers"].max()) * 1.18], title=None)
        fig.update_xaxes(title=None)
        polish_chart(fig, 380)

    with right:
        st.markdown("### Contract mix")
        if contract_col:
            contract_mix = (
                customers_df[contract_col]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Contract")
                .to_frame("Customers")
            )
            contract_plot = contract_mix.reset_index()
            contract_plot.columns = ["Contract", "Customers"]
            fig = px.bar(
                contract_plot, x="Contract", y="Customers",
                color="Contract", color_discrete_sequence=CHART_COLORS,
                text="Customers",
            )
            fig.update_traces(marker_line_width=0, texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_yaxes(range=[0, float(contract_plot["Customers"].max()) * 1.18], title=None)
            fig.update_xaxes(title=None)
            polish_chart(fig, 380)

    st.markdown("### Customer value summary")
    if clv_df is not None:
        st.dataframe(clv_df, use_container_width=True, hide_index=True)
    else:
        st.info("Add clv_summary.csv beside app.py.")

# -------------------- Churn Analysis --------------------
elif page == "Churn Analysis":
    st.subheader("Churn analysis")

    a, b = st.columns(2)
    a.metric("Churned customers", f"{churned_customers:,}")
    b.metric("Observed churn rate", f"{churn_rate:.1%}")

    st.markdown("### Churn by contract")
    if contract_col:
        contract_churn = (
            customers_df.groupby(contract_col, dropna=False)["_churned"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        contract_plot = contract_churn.rename("Churn rate").reset_index()
        contract_plot.columns = ["Contract", "Churn rate"]
        fig = px.bar(
            contract_plot.sort_values("Churn rate"), x="Churn rate", y="Contract",
            orientation="h", color="Churn rate",
            color_continuous_scale=["#B8DCE5", "#168EAD", "#6546D7"],
            text=contract_plot.sort_values("Churn rate")["Churn rate"].map(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(marker_line_width=0, textposition="outside")
        fig.update_xaxes(title=None, ticksuffix="%", range=[0, float(contract_plot["Churn rate"].max()) * 1.22], color="#66758F")
        fig.update_yaxes(title=None)
        polish_chart(fig, 390)

    st.markdown("### Saved SHAP feature importance")
    if shap_df is not None:
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

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
                color="Importance", color_continuous_scale=["#B8DCE5", "#168EAD", "#6546D7"],
            )
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            polish_chart(fig, 440)

    st.markdown("### Model evaluation")

    metrics_df = load_csv("model_evaluation.csv")
    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Add model_evaluation.csv with actual model metrics.")

    confusion_df = load_csv("confusion_matrix.csv")
    if confusion_df is not None:
        st.markdown("### Confusion matrix")
        st.dataframe(confusion_df, use_container_width=True, hide_index=True)
    else:
        st.info("Add confusion_matrix.csv with actual confusion matrix values.")

# -------------------- Customer Insights --------------------
elif page == "Customer Insights":
    st.subheader("Customer insights")

    if cluster_df is not None:
        st.markdown("### Customer cluster profiles")
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    if contract_df is not None:
        st.markdown("### Contract retention summary")
        st.dataframe(contract_df, use_container_width=True, hide_index=True)

    if clv_df is not None:
        st.markdown("### CLV summary")
        st.dataframe(clv_df, use_container_width=True, hide_index=True)

    st.markdown("### Browse customer records")
    if customer_id_col:
        customer_options = customers_df[customer_id_col].astype(str).tolist()
        selected_customer = st.selectbox("Select a customer", customer_options)

        customer_row = customers_df[
            customers_df[customer_id_col].astype(str) == selected_customer
        ].iloc[0]

        st.dataframe(
            customer_row.drop(labels=["_churned"]).to_frame("Value"),
            use_container_width=True,
        )
    else:
        st.dataframe(customers_df.head(100), use_container_width=True)

# -------------------- Retention Campaigns --------------------
elif page == "Retention Campaigns":
    st.subheader("Retention campaigns")

       # ---------- Customer-level churn risk ----------
    st.markdown("### Customer-level churn risk")

    risk_df = load_csv("customer_churn_risk.csv")

    if risk_df is not None:
        risk_df.columns = [
            str(col).strip().lower() for col in risk_df.columns
        ]

        id_col = find_col(risk_df, ["customerid", "customer_id", "customer"])
        probability_col = find_col(
            risk_df, ["churn_probability", "churn_prob", "probability"]
        )
        level_col = find_col(risk_df, ["risk_level"])
        focus_col = find_col(risk_df, ["contact_focus"])

        required_cols = [id_col, probability_col, level_col, focus_col]

        if any(col is None for col in required_cols):
            st.error(
                "customer_churn_risk.csv must contain customerID, "
                "churn_probability, risk_level, and contact_focus."
            )
        else:
            risk_df[probability_col] = pd.to_numeric(
                risk_df[probability_col], errors="coerce"
            )
            risk_df = risk_df.dropna(subset=[probability_col])
            risk_df = risk_df.sort_values(
                probability_col, ascending=False
            )

            st.caption(
                "Sorted by model-estimated churn probability. "
                "These are risk estimates, not guaranteed churn outcomes "
                "or proof that a retention action will work."
            )

            display_cols = [
                id_col, probability_col, level_col, focus_col
            ]

            selected_risk = st.selectbox(
                "Filter by risk level",
                ["All", "High", "Medium", "Low"],
            )

            filtered_risk_df = (
                risk_df
                if selected_risk == "All"
                else risk_df[
                    risk_df[level_col].astype(str).str.title() == selected_risk
                ]
            )

            # Keep short-value columns compact; give the long contact guidance
            # a controlled width instead of letting it dominate the table.
            risk_column_config = {
                id_col: st.column_config.TextColumn(
                    "Customer ID", width="small"
                ),
                probability_col: st.column_config.NumberColumn(
                    "Churn probability", width="small", format="percent"
                ),
                level_col: st.column_config.TextColumn(
                    "Risk level", width="small"
                ),
                focus_col: st.column_config.TextColumn(
                    "Contact focus", width="large"
                ),
            }

            st.dataframe(
                filtered_risk_df[display_cols],
                column_config=risk_column_config,
                use_container_width=True,
                hide_index=True,
                height=390,
            )

            st.download_button(
                "Download customer churn risk",
                risk_df.to_csv(index=False),
                "customer_churn_risk.csv",
                "text/csv",
            )
    else:
        st.info(
            "Place customer_churn_risk.csv beside app.py. "
            "Export it from Colab and copy it into the app folder."
        )

    # ---------- Kaplan-Meier survival analysis ----------
    st.markdown("### Kaplan–Meier survival analysis")

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test

        if tenure_col is None:
            st.warning("The customer table has no tenure column.")
        else:
            survival_df = customers_df[[tenure_col, churn_col]].copy()
            survival_df[tenure_col] = pd.to_numeric(
                survival_df[tenure_col], errors="coerce"
            )
            survival_df = survival_df.dropna(subset=[tenure_col])
            survival_df = survival_df[survival_df[tenure_col] >= 0]

            survival_df["_event"] = (
                survival_df[churn_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["yes", "true", "1", "churned"])
                .astype(int)
            )

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=survival_df[tenure_col],
                event_observed=survival_df["_event"],
                label="All customers"
            )

            survival_curve = kmf.survival_function_.reset_index()
            survival_curve.columns = ["Tenure (months)", "Churn-free probability"]

            fig = px.line(
                survival_curve, x="Tenure (months)", y="Churn-free probability",
                color_discrete_sequence=["#6546D7"],
            )
            fig.update_traces(line=dict(width=3), hovertemplate="Month %{x}<br>Churn-free: %{y:.1%}<extra></extra>")
            fig.update_yaxes(title=None, tickformat=".0%", range=[0, 1.02])
            fig.update_xaxes(title="Tenure (months)")
            polish_chart(fig, 400)

            m1, m2 = st.columns(2)
            m1.metric(
                "12-month churn-free estimate",
                f"{kmf.predict(12) * 100:.2f}%"
            )
            m2.metric(
                "24-month churn-free estimate",
                f"{kmf.predict(24) * 100:.2f}%"
            )

            st.caption(
                "Kaplan–Meier estimates the probability of remaining churn-free "
                "over tenure, accounting for customers who have not churned yet."
            )

            if contract_col:
                st.markdown("#### Survival by contract type")

                contract_curves = pd.DataFrame()

                for contract_name in survival_df[contract_col].dropna().unique() \
                        if contract_col in survival_df.columns else []:
                    pass

                # Build contract-level curves using the original customer table.
                contract_survival = customers_df[
                    [tenure_col, churn_col, contract_col]
                ].copy()
                contract_survival[tenure_col] = pd.to_numeric(
                    contract_survival[tenure_col], errors="coerce"
                )
                contract_survival = contract_survival.dropna(
                    subset=[tenure_col, contract_col]
                )

                curve_frames = []
                for contract_name, group in contract_survival.groupby(contract_col):
                    event = (
                        group[churn_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["yes", "true", "1", "churned"])
                        .astype(int)
                    )

                    group_kmf = KaplanMeierFitter()
                    group_kmf.fit(
                        durations=group[tenure_col],
                        event_observed=event,
                        label=str(contract_name)
                    )

                    curve = group_kmf.survival_function_.reset_index()
                    curve.columns = ["Tenure (months)", str(contract_name)]
                    curve_frames.append(curve)

                if curve_frames:
                    combined = curve_frames[0]
                    for curve in curve_frames[1:]:
                        combined = combined.merge(
                            curve,
                            on="Tenure (months)",
                            how="outer"
                        )

                    combined_long = combined.melt(
                        id_vars=["Tenure (months)"], var_name="Contract",
                        value_name="Churn-free probability"
                    ).dropna()
                    fig = px.line(
                        combined_long, x="Tenure (months)", y="Churn-free probability",
                        color="Contract", color_discrete_sequence=CHART_COLORS,
                    )
                    fig.update_traces(line=dict(width=3), hovertemplate="%{fullData.name}<br>Month %{x}<br>Churn-free: %{y:.1%}<extra></extra>")
                    fig.update_yaxes(title=None, tickformat=".0%", range=[0, 1.02])
                    fig.update_xaxes(title="Tenure (months)")
                    polish_chart(fig, 420)

                
            
            try:
                logrank = multivariate_logrank_test(
                    event_durations=contract_survival[tenure_col],
                    groups=contract_survival[contract_col],
                    event_observed=(
                        contract_survival[churn_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["yes", "true", "1", "churned"])
                        .astype(int)
                    )
                )

                p_value = logrank.p_value

                if p_value < 0.0001:
                    st.write("Contract-group log-rank test p-value: < 0.0001")
                else:
                    st.write(
                        f"Contract-group log-rank test p-value: {p_value:.4f}"
                    )

            except Exception as error:
                st.caption(
                    f"Could not calculate the log-rank test: {error}"
                )



    except ImportError:
        st.warning(
            "Kaplan–Meier and Cox analysis require lifelines. "
            "Install it with: pip install lifelines"
        )

    # ---------- Cox proportional-hazards model ----------
    st.markdown("### Cox proportional-hazards model")

    try:
        from lifelines import CoxPHFitter

        if tenure_col is None:
            st.warning("The customer table has no tenure column.")
        else:
            cox_df = customers_df.copy()

            cox_df["_duration"] = pd.to_numeric(
                cox_df[tenure_col], errors="coerce"
            )
            cox_df["_event"] = (
                cox_df[churn_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["yes", "true", "1", "churned"])
                .astype(int)
            )

            # Keep a small set of interpretable customer features.
            candidate_features = [
                "tenure",
                "monthly_charges",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "Contract",
                "InternetService",
                "TechSupport",
                "OnlineSecurity",
                "PaymentMethod"
            ]
            feature_cols = [
                col for col in candidate_features
                if col in cox_df.columns and col != tenure_col
            ]

            if not feature_cols:
                st.warning(
                    "No eligible customer features were found for the Cox model."
                )
            else:
                model_df = cox_df[
                    ["_duration", "_event"] + feature_cols
                ].copy()

                model_df = model_df.dropna(
                    subset=["_duration", "_event"]
                )
                model_df = model_df[model_df["_duration"] > 0]

                for col in feature_cols:
                    if model_df[col].dtype == "object":
                        model_df[col] = model_df[col].fillna("Unknown")
                    else:
                        model_df[col] = pd.to_numeric(
                            model_df[col], errors="coerce"
                        )

                model_df = model_df.dropna()

                # One-hot encode categorical features and remove one
                # reference category per feature.
                model_df = pd.get_dummies(
                    model_df,
                    columns=[
                        col for col in feature_cols
                        if model_df[col].dtype == "object"
                    ],
                    drop_first=True,
                    dtype=float
                )

                # Remove constant columns, which cannot be estimated.
                predictors = [
                    col for col in model_df.columns
                    if col not in ["_duration", "_event"]
                    and model_df[col].nunique() > 1
                ]

                model_df = model_df[
                    ["_duration", "_event"] + predictors
                ]

                if not predictors:
                    st.warning("The selected features have no variation.")
                else:
                    cph = CoxPHFitter(penalizer=0.01)
                    cph.fit(
                        model_df,
                        duration_col="_duration",
                        event_col="_event"
                    )

                    st.caption(
                        "Hazard ratio above 1 indicates a higher estimated churn "
                        "hazard; below 1 indicates a lower estimated hazard. "
                        "These are associations, not causal effects."
                    )

                    cox_results = cph.summary.reset_index()
                    cox_results = cox_results.rename(
                        columns={
                            "covariate": "Feature",
                            "exp(coef)": "Hazard ratio",
                            "exp(coef) lower 95%": "HR 95% lower",
                            "exp(coef) upper 95%": "HR 95% upper",
                            "p": "p-value"
                        }
                    )

                    display_cols = [
                        col for col in [
                            "Feature",
                            "Hazard ratio",
                            "HR 95% lower",
                            "HR 95% upper",
                            "p-value"
                        ]
                        if col in cox_results.columns
                    ]

                    st.dataframe(
                        cox_results[display_cols].round(4),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download Cox model results",
                        cox_results.to_csv(index=False),
                        "cox_model_results.csv",
                        "text/csv"
                    )

    except ImportError:
        st.warning(
            "Cox model analysis requires lifelines. "
            "Install it with: pip install lifelines"
        )
    except Exception as error:
        st.warning(f"Cox model could not be fitted: {error}")

    # ---------- Existing saved campaign scenario ----------
    st.markdown("### Saved campaign scenario")

    if roi_scenario_df is not None:
        st.dataframe(
            roi_scenario_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Add retention_roi_scenario.csv beside app.py.")
# -------------------- ROI Simulator --------------------
elif page == "Retention ROI Simulator":
    st.subheader("Retention ROI Simulator")

    st.caption(
        "Adjust campaign assumptions. Results are modeled estimates, not guaranteed savings."
    )

    with st.sidebar:
        st.markdown("### Campaign settings")

        segment = st.selectbox(
            "Customer group",
            ["All customers"] + (
                sorted(customers_df[contract_col].dropna().astype(str).unique().tolist())
                if contract_col else []
            ),
        )

        if segment == "All customers" or not contract_col:
            segment_df = customers_df
        else:
            segment_df = customers_df[
                customers_df[contract_col].astype(str) == segment
            ]

        segment_count = len(segment_df)

        segment_avg_charge = (
            float(segment_df[monthly_col].mean())
            if monthly_col and not segment_df.empty
            else avg_monthly_charge
        )

        target_count = st.slider(
            "Customers targeted",
            min_value=1,
            max_value=max(1, segment_count),
            value=min(1000, max(1, segment_count)),
        )

        discount_pct = st.slider("Discount offered", 0, 50, 10)
        discount_months = st.slider("Discount duration", 1, 12, 3)

        churn_reduction_pct = st.slider(
            "Assumed churn reduction",
            1,
            20,
            5,
        )

        extra_months = st.slider("Extra months retained", 1, 24, 6)

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue preserved", money(revenue_preserved))
    k2.metric("Discount cost", money(discount_cost))
    k3.metric("Net benefit", money(net_benefit))
    k4.metric("Scenario ROI", f"{roi:.1f}%")

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
            color_discrete_map={"Revenue preserved": "#168EAD", "Discount cost": "#F08A5D"},
            text="Amount ($)",
        )
        fig.update_traces(marker_line_width=0, texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_xaxes(title=None, tickprefix="$")
        fig.update_yaxes(title=None)
        polish_chart(fig, 360)

    with right:
        st.markdown("### Scenario details")
        st.metric("Customers in selected group", f"{segment_count:,}")
        st.metric("Estimated additional retained", f"{additional_retained:,.0f}")
        st.metric("Group average monthly charge", money(segment_avg_charge))

        if net_benefit > 0:
            st.success("Estimated revenue preserved exceeds discount cost.")
        elif net_benefit < 0:
            st.warning("Estimated discount cost exceeds revenue preserved.")
        else:
            st.info("Estimated revenue preserved equals discount cost.")

    st.info(
        "The customer group and average charge come from PostgreSQL. "
        "Churn reduction and extra retention period are assumptions. "
        "This is a scenario calculator, not a measured campaign result."
    )

    sensitivity_path = APP_DIR / "retention_roi_sensitivity.png"
    if sensitivity_path.exists():
        st.markdown("### Saved sensitivity analysis")
        st.image(
            str(sensitivity_path),
            caption="Sensitivity analysis exported from the project notebook.",
            use_container_width=True,
        )
