
import os
from pathlib import Path
import pandas as pd
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
    path = APP_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def find_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def money(value):
    return f"${value:,.2f}"


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

# -------------------- Header and navigation --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f8ff 0%, #edf3fc 100%);
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5eaf3;
    }
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        padding: 1.7rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #2563eb);
        color: white;
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white;
        margin: .3rem 0 .5rem 0;
        font-size: 2.1rem;
    }
    .hero p {
        color: #dbeafe;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf3;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, .04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div style="font-size:.8rem;font-weight:700;letter-spacing:.12em;">
            CUSTOMER ANALYTICS • RETENTION MANAGEMENT
        </div>
        <h1>Customer Intelligence</h1>
        <p>Explore churn, customer behavior, and retention campaign scenarios.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Churn Analysis",
        "Customer Insights",
        "Retention Campaigns",
        "Retention ROI Simulator",
    ],
)

st.sidebar.caption(f"Connected records: {total_customers:,}")

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
        st.bar_chart(churn_summary)

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
            st.bar_chart(contract_mix)

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
        st.bar_chart(contract_churn)

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
            st.bar_chart(plot_df.head(15))

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

            st.dataframe(
                filtered_risk_df[display_cols],
                use_container_width=True,
                hide_index=True,
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

            st.line_chart(
                survival_curve.set_index("Tenure (months)")
            )

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

                    st.line_chart(
                        combined.set_index("Tenure (months)")
                    )

                try:
                    logrank = multivariate_logrank_test(
    contract_survival[tenure_col],
    contract_survival[contract_col],
    event_durations=contract_survival[tenure_col],
    event_observed=(
        contract_survival[churn_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1", "churned"])
        .astype(int)
    ),
)
                    st.write(
                        f"Contract-group log-rank test p-value: "
                        f"{logrank.p_value:.4g}"
                    )
                except Exception as error:
                    st.caption(f"Could not calculate the log-rank test: {error}")

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
        st.bar_chart(
            pd.DataFrame(
                {"Amount ($)": [revenue_preserved, discount_cost]},
                index=["Revenue preserved", "Discount cost"],
            ),
            horizontal=True,
        )

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
