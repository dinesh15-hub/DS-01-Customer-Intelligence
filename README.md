# Customer Lifetime Value & Churn Intelligence System

An end-to-end customer analytics project built to understand why customers leave, identify customers at risk of churning, and explore ways businesses can improve customer retention.

This project was developed as part of the **DS-01 Internship Project at Nexvia Technologies** using the IBM Telco Customer Churn dataset.

## About the Project

Customer churn is a major challenge for subscription-based businesses. Losing customers affects recurring revenue, and understanding the reasons behind those decisions can help businesses make better retention strategies.

The goal of this project was to build a single platform that brings together customer churn analysis, predictive modelling, customer segmentation, lifetime value analysis, and retention planning.

I used Python for data analysis and machine learning, PostgreSQL for storing customer records, and Streamlit to bring the results together in an interactive dashboard.

The project uses **7,043 customer records** from the IBM Telco Customer Churn dataset.

## What the Project Covers

The application is organised into five main sections:

### 1. Business Overview
- Displays the total customer count and overall churn rate.
- Provides an overview of customer distribution and contract types.
- Highlights important business-level metrics.

### 2. Churn Analysis
- Examines customer churn across different contract types and service categories.
- Uses LightGBM to predict customer churn.
- Applies SHAP to understand which features contribute most to model predictions.
- Helps identify customer characteristics associated with higher churn risk.

### 3. Customer Insights
- Groups customers into different segments.
- Explores customer behaviour and revenue patterns.
- Estimates customer lifetime value using historical revenue information.

### 4. Retention Campaigns
- Displays customer-level churn risk estimates.
- Provides customer-focused retention suggestions.
- Uses Kaplan–Meier survival analysis to study customer retention over time.
- Applies a Cox Proportional Hazards model to examine factors associated with churn.

### 5. Retention ROI Simulator
- Allows users to explore different retention campaign scenarios.
- Estimates potential revenue preservation and campaign costs.
- Helps compare possible outcomes under different assumptions.

## Tools & Technologies

| Technology | Purpose |
|---|---|
| Python | Data analysis and application development |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | Model preparation and evaluation |
| LightGBM | Churn prediction |
| SHAP | Model explainability |
| Lifelines | Survival analysis |
| PostgreSQL (Neon) | Customer data storage |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts and visualisations |

## Dataset

**Source:** IBM Telco Customer Churn dataset

- Total records: 7,043
- Original features: 21 predictors and 1 target column
- Target variable: `Churn`

The dataset contains customer demographics, subscribed services, contract details, payment information, tenure, and billing information.

Data cleaning and preprocessing were performed before using the dataset for analysis and modelling.

## Machine Learning & Analysis

The project includes the following analytical work:

- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Churn classification using Logistic Regression and LightGBM
- Model evaluation and comparison
- SHAP-based feature importance analysis
- Customer segmentation and clustering
- Customer lifetime value estimation
- Kaplan–Meier survival analysis
- Cox Proportional Hazards modelling
- Retention campaign ROI scenario analysis

The aim was not just to generate predictions, but also to make the results understandable from a business perspective.

## Application & Database

Customer records are stored in a Neon PostgreSQL database and accessed by the Streamlit application.

The dashboard currently contains five functional pages, with interactive charts, customer-level risk information, and scenario-based analysis.

**Live Dashboard:**  
[Customer Intelligence – Streamlit App](https://ds-01-customer-intelligence-project.streamlit.app/)

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/dinesh15-hub/DS-01-Customer-Intelligence.git
cd DS-01-Customer-Intelligence
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure PostgreSQL credentials

Create a file named `secrets.toml` inside the `.streamlit` folder.

Add your own PostgreSQL connection details:

```toml
[postgres]
host = "YOUR_HOST"
port = 5432
user = "YOUR_USERNAME"
password = "YOUR_PASSWORD"
database = "YOUR_DATABASE"
sslmode = "require"
```

Make sure the credentials are kept private and are never committed to GitHub.

### 5. Start the application

```bash
streamlit run app.py
```

The application should open in your browser at:

`http://localhost:8501`

## Current Project Status

The main customer analytics application has been implemented and deployed.

Completed work includes data preparation, churn prediction, model explainability, customer segmentation, CLV estimation, survival analysis, PostgreSQL integration, and the interactive ROI simulator.

**Uplift modelling:** Customer-level uplift outputs are present in the project workflow, but the results still require validation. They should not be interpreted as proven causal campaign effects.

## What I Learned

Working on this project gave me practical experience in taking a dataset through the complete analytics workflow, from cleaning and exploration to model development and dashboard deployment.

I also got to work with PostgreSQL, build interactive visualisations, interpret machine learning results, and connect analytical findings to business decisions.

One of the most important takeaways was that a useful analytics solution should do more than display numbers. It should help someone understand what is happening and what they could do next.

## Author

**DS-01 Internship Project**  
Nexvia Technologies

**GitHub:** [dinesh15-hub](https://github.com/dinesh15-hub)
