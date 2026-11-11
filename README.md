# Customer Lifetime Value & Churn Intelligence System

**DS-01 Internship Project | Nexvia Technologies**

An end-to-end customer analytics project built using the IBM Telco Customer Churn dataset to understand customer behaviour, predict churn, estimate customer lifetime value, and explore retention strategies.

**Live Dashboard:** [Customer Intelligence – Streamlit](https://ds-01-customer-intelligence-project.streamlit.app/)

**GitHub:** [DS-01 Customer Intelligence](https://github.com/dinesh15-hub/DS-01-Customer-Intelligence)

## About the Project

Customer churn is a major challenge for subscription-based businesses. Understanding why customers leave and identifying those at risk can help businesses improve retention and protect recurring revenue.

The goal of this project was to develop a customer intelligence platform combining data analysis, machine learning, survival analysis, and business insights in one interactive dashboard.

The project uses **7,043 customer records** from the IBM Telco Customer Churn dataset. The workflow covers data cleaning, exploratory analysis, churn prediction, customer segmentation, CLV estimation, and retention ROI analysis.

## Dashboard Features

The Streamlit application includes five main sections:

### 1. Business Overview
- Total customers, churn rate, and monthly revenue.
- Customer distribution and contract mix.
- Key business performance metrics.

### 2. Churn Analysis
- Churn patterns across contracts and services.
- Churn prediction using Logistic Regression and LightGBM.
- Model evaluation and SHAP feature importance.

### 3. Customer Insights
- Customer segmentation using clustering.
- Customer behaviour and revenue analysis.
- Customer lifetime value estimation.

### 4. Retention Campaigns
- Customer-level churn risk predictions.
- Risk categorisation and retention suggestions.
- Kaplan–Meier survival analysis.
- Cox Proportional Hazards modelling.

### 5. Retention ROI Simulator
- Interactive retention campaign scenarios.
- Estimated revenue preservation and campaign costs.
- Comparison of potential campaign outcomes.

## Dataset

**Source:** IBM Telco Customer Churn dataset

- Customer records: 7,043
- Original columns: 22
- Target variable: `Churn`

The dataset includes customer demographics, tenure, subscribed services, contract details, payment methods, and billing information.

Data cleaning and feature engineering were performed before analysis and model development.

## Technologies Used

- **Python** – Main programming language
- **Pandas & NumPy** – Data cleaning and manipulation
- **Scikit-learn** – Machine learning and preprocessing
- **LightGBM** – Churn classification
- **SHAP** – Model explainability
- **Lifelines** – Survival analysis
- **PostgreSQL & Neon** – Customer data storage
- **Streamlit** – Dashboard development
- **Plotly** – Interactive visualisations

## Key Analysis

The project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Logistic Regression and LightGBM churn models
- Model evaluation and comparison
- SHAP-based feature importance
- Customer clustering and segmentation
- Customer lifetime value estimation
- Kaplan–Meier survival analysis
- Cox Proportional Hazards modelling
- Retention ROI scenario analysis

## Uplift Modelling – Limitation

Uplift modelling using CausalML was part of the original requirements. However, the IBM Telco dataset does not contain customer treatment assignments or retention offer-response data. Without this information, the actual impact of retention campaigns cannot be reliably measured. Therefore, uplift modelling remains unimplemented.

## Running the Project Locally

**1. Clone the repository**

```bash
git clone https://github.com/dinesh15-hub/DS-01-Customer-Intelligence.git
cd DS-01-Customer-Intelligence
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure PostgreSQL credentials**

Create `.streamlit/secrets.toml` and add your own database credentials:

```toml
[postgres]
host = "YOUR_HOST"
port = 5432
user = "YOUR_USERNAME"
password = "YOUR_PASSWORD"
database = "YOUR_DATABASE"
sslmode = "require"
```

Keep your credentials private. Never upload `secrets.toml` to GitHub.

**5. Run the application**

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## What I Learned

Working on this project gave me practical experience with the complete data analytics workflow, from preparing raw data and building machine learning models to interpreting results and deploying an interactive application.

It also helped me understand how analytical findings can be connected to real business problems, particularly customer retention and revenue management.

## Acknowledgement

Developed as part of the **DS-01 Internship Project at Nexvia Technologies**.

Dataset: IBM Telco Customer Churn.

## Author

**Dinesh Chandra Reddy**

[GitHub Profile](https://github.com/dinesh15-hub)
