# Bank Loan Underwriting & Risk Prediction AI

## Business Problem
A retail bank is facing elevated non-performing assets (NPAs). Management needs to identify applicants likely to default so the bank can reduce financial losses without slowing down approval turnaround times.

## Objective
Develop a classification machine learning model that predicts whether a loan applicant is likely to default (High Risk) or qualify (Low Risk), and deploy it as an interactive decision-support tool.

## Dataset
* **Name:** Loan Prediction Dataset
* **Source:** Kaggle
* **Size:** 614 rows, 13 columns

## Methodology
1. **Data Cleaning:** Removed irrelevant IDs, imputed missing numerical values with medians, and categorical values with modes.
2. **Exploratory Data Analysis (EDA):** Analyzed approval rates across credit history and property areas.
3. **Feature Preparation:** Encoded categorical variables into numerical formats.
4. **Train/Test Split:** Split data 80/20.
5. **Model Training:** Trained a Random Forest Classifier using balanced class weights.
6. **Model Saving:** Exported the trained model using `joblib` for deployment.
7. **Streamlit App & Deployment:** Built a 3-page interactive app and deployed it via Streamlit Community Cloud.

## Model Used & Performance
* **Algorithm:** Random Forest Classifier
* **Primary Metric:** Precision (Chosen because the financial cost of a False Positive—approving a bad loan—is significantly higher than a False Negative).

## Key Business Insights
1. **Credit History is King:** Applicants with a 0.0 credit history are almost universally rejected.
2. **Semi-Urban Profitability:** The semi-urban segment has the highest volume of safe applications.

## How to Run the Application Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Live Links
* **Deployed Application:** [https://angry-facts-rescue.loca.lt/]