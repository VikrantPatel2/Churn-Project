# Customer Intelligence & Churn Prediction System

An end-to-end customer analytics project built to explore customer
behavior, predict telecom customer churn, explain model predictions, and
present insights through an interactive dashboard and API.

> **Project status:** The core analysis and modeling workflow has been
> developed in stages. FastAPI should be tested in the local environment
> before claiming the API is operational. Docker deployment is
> intentionally deferred.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Dataset](#dataset)
-   [Workflow](#workflow)
-   [Setup](#setup)
-   [Run the Streamlit Dashboard](#run-the-streamlit-dashboard)
-   [Run the FastAPI Service](#run-the-fastapi-service)
-   [Model Artifacts](#model-artifacts)
-   [Evaluation and Explainability](#evaluation-and-explainability)
-   [Current Limitations](#current-limitations)
-   [Future Improvements](#future-improvements)

## Project Overview

Customer churn occurs when a customer stops using a company's service.
Retaining customers is important for subscription-based businesses,
including telecommunications providers.

This project uses customer data to analyze churn patterns and build a
machine-learning pipeline that estimates the likelihood that a customer
will leave. The project also explores customer segmentation, model
interpretation, retrieval over generated project reports, and deployment
options.

## Objectives

-   Explore customer characteristics and churn patterns using data
    analysis.
-   Clean and prepare data for machine learning.
-   Engineer and preprocess features.
-   Train and compare classification models.
-   Tune model hyperparameters and evaluate classification performance.
-   Examine prediction thresholds and customer risk segments.
-   Explain model behavior using SHAP.
-   Build an interactive Streamlit dashboard.
-   Expose predictions through a FastAPI service.
-   Demonstrate data drift monitoring concepts.

## Technology Stack

-   **Language:** Python
-   **Data analysis:** pandas, NumPy
-   **Visualization:** Matplotlib and other notebook visualization
    libraries used in the project
-   **Machine learning:** scikit-learn
-   **Model persistence:** joblib
-   **Explainability:** SHAP
-   **Dashboard:** Streamlit
-   **API:** FastAPI, Uvicorn, Pydantic
-   **Database / SQL practice:** SQLite
-   **Development environment:** Jupyter Notebook
-   **Containerization:** Docker planned, but currently deferred

Install only the packages needed for the components you intend to run.
Your saved model may require the same or compatible scikit-learn version
as the one used during training.

## Project Structure

The exact files may vary depending on your local folder. A typical
layout is:

``` text
Project/
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── customer_intelligence.db
├── models/
│   ├── customer_churn_optimized_pipeline.joblib
│   └── customer_churn_config.joblib
├── reports/
│   └── generated analysis and evaluation CSV files
├── notebooks/
│   └── analysis, modeling, NLP, drift, and explainability notebooks
├── app.py
├── api.py
├── requirements.txt
└── README.md
```

Your notebook files may be stored directly in the project root instead
of a `notebooks/` directory. Keep the model paths in `app.py` and
`api.py` consistent with the actual locations on your computer.

## Dataset

The project uses the IBM Telco Customer Churn dataset. It contains
customer demographics, subscribed services, account information, billing
attributes, and a churn label.

Place the CSV file in the `data/` directory if your notebooks expect
this path:

``` text
data/Telco-Customer-Churn.csv
```

The dataset may contain customer information. Use it responsibly, avoid
publishing sensitive or identifying records, and follow the dataset's
applicable terms.

## Workflow

1.  **Data preparation:** Load the dataset, inspect data types and
    missing values, and prepare columns for analysis.
2.  **Exploratory data analysis:** Study churn distribution and
    relationships between customer attributes and churn.
3.  **SQL analytics:** Use SQLite to practice customer-level and
    aggregate queries.
4.  **Feature engineering:** Prepare numerical and categorical features
    and create preprocessing pipelines.
5.  **Model training and evaluation:** Compare classification approaches
    such as logistic regression, decision trees, random forests, and
    boosting models.
6.  **Model optimization:** Use cross-validation and hyperparameter
    tuning, then compare metrics such as precision, recall, F1-score,
    and ROC-AUC where applicable.
7.  **Threshold and risk analysis:** Explore how changing the
    classification threshold affects predicted churn and risk groups.
8.  **Retrieval-based report assistant:** Retrieve relevant information
    from generated report files. This is retrieval over project
    artifacts, not necessarily a full LLM-based RAG system.
9.  **Prediction interface:** Load the saved pipeline to score customer
    records.
10. **Streamlit dashboard:** Present customer insights and model
    predictions in an interactive interface.
11. **Data drift demonstration:** Explore changes in input data
    distributions and why monitoring matters.
12. **Explainability:** Use SHAP to investigate feature contributions,
    subject to the model and explainer configuration.
13. **FastAPI service:** Provide HTTP endpoints for health checks and
    customer predictions.
14. **Docker:** Containerization instructions have been prepared, but
    Docker is currently skipped because it is not working in the local
    environment.

## Setup

### 1. Open the project directory

In PowerShell, change to your actual project directory. For example:

``` powershell
cd C:\Users\vikra\Desktop\Project
```

### 2. Create and activate a virtual environment (optional but recommended)

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use your usual Python environment or
follow your organization's policy for script execution.

### 3. Install dependencies

Create a `requirements.txt` file containing the libraries needed by your
application. A minimal starting point for the dashboard and API is:

``` text
pandas
numpy
scikit-learn
joblib
streamlit
fastapi
uvicorn[standard]
pydantic
```

Add notebook, plotting, and explainability libraries such as `jupyter`,
`matplotlib`, and `shap` if your notebooks require them. For
reproducibility, pin compatible versions after confirming the versions
used to train the model.

Install the dependencies:

``` powershell
python -m pip install -r requirements.txt
```

### 4. Check the saved model files

The prediction application may expect:

``` text
models/customer_churn_optimized_pipeline.joblib
models/customer_churn_config.joblib
```

Confirm the paths and filenames match the loading code in your
application.

## Run the Streamlit Dashboard

From the project directory, run:

``` powershell
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal, commonly
`http://localhost:8501`. Open that URL in your browser.

If the dashboard cannot find the model, check the model paths in
`app.py` and make sure the model files exist.

## Run the FastAPI Service

From the project directory:

``` powershell
python -m uvicorn api:app --reload
```

If your project is in `C:\Users\vikra\Desktop\Project`, run `cd` to that
directory first. Running Uvicorn from another directory can cause an
import error such as `Could not import module "api"`.

When the service starts successfully, open:

-   Interactive API docs: `http://127.0.0.1:8000/docs`
-   Health endpoint: `http://127.0.0.1:8000/health`

The health endpoint must exist in `api.py` for that URL to work. Use the
actual route names defined in your API code. Test a prediction using the
request schema displayed in `/docs`.

**Note:** The API should be considered verified only after the service
starts and a test prediction returns the expected response.

## Model Artifacts

The optimized model pipeline is expected to be stored at:

``` text
models/customer_churn_optimized_pipeline.joblib
```

Configuration, such as the selected classification threshold, may be
stored at:

``` text
models/customer_churn_config.joblib
```

Load joblib artifacts only from trusted sources. A model serialized with
joblib/pickle can execute code during loading. Scikit-learn artifacts
are also not guaranteed to be compatible across library versions.

## Evaluation and Explainability

The modeling workflow covers commonly used classification measures:

-   **Precision:** Of the customers predicted to churn, the proportion
    that actually churn.
-   **Recall:** Of the customers who actually churn, the proportion
    detected by the model.
-   **F1-score:** The harmonic mean of precision and recall.
-   **ROC-AUC:** A threshold-independent measure of how well the model
    ranks positive and negative examples, when applicable.
-   **Cross-validation:** Repeated evaluation across data folds to
    estimate generalization performance.

For churn prediction, recall can be important when missing a likely
churner is costly, but increasing recall may also increase false
positives. Select a threshold based on the business objective and
validation results rather than assuming 0.5 is always optimal.

SHAP can help explain model outputs by estimating feature contributions.
Explanations describe model behavior and should not automatically be
interpreted as causal effects.

No specific model score is listed here because verified final evaluation
values have not been supplied in this README.

## Current Limitations

-   Docker build and container execution are deferred.
-   FastAPI must be locally tested before describing it as operational.
-   Exact notebook filenames and report filenames may differ from the
    example structure.
-   Model performance depends on the dataset, preprocessing, validation
    method, and decision threshold.
-   A retrieval-based assistant over CSV reports is not the same as a
    complete grounded LLM/RAG system.
-   The project is an analytics and prediction prototype, not a
    production-ready customer decision system.

## Future Improvements

-   Verify and document final cross-validation and holdout metrics.
-   Add automated tests for preprocessing, predictions, and API
    validation.
-   Add input validation, logging, and exception handling to the API.
-   Integrate the Streamlit dashboard with the FastAPI service if
    desired.
-   Add monitoring for data quality, prediction distributions, and model
    performance.
-   Add authentication and privacy safeguards before exposing an API
    publicly.
-   Pin dependency versions and document reproducible setup steps.
-   Revisit Docker once the local Docker environment is working.
-   Deploy the API and dashboard to an appropriate hosting platform
    after testing.

## Author
Vikrant Patel
**Project:** Customer Intelligence & Churn Prediction System

