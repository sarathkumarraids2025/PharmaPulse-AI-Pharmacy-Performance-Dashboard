# Customer Churn Analytics & Business KPI Dashboard

An interactive Streamlit dashboard that turns the cleaned customer churn data from Task 2 into decision-ready KPIs and charts covering churn, retention, revenue at risk and customer segments.

**Task:** EdVyro Data Analytics, Task 3 - Business KPI Dashboard.
**Live dashboard:** _paste your Streamlit Community Cloud URL here after deploying._

> **Small sample warning:** the dataset has only 15 customers. Percentages move a lot with a single customer, so results must not be generalised to a larger population.

## Features
- Four KPI cards (churn rate, retention rate, monthly revenue at risk, average tenure) that recalculate from the filtered data
- Sidebar filters for contract type, tenure range and payment method, which work together, plus a Reset button
- Six working pages: Overview, Churn Analysis, Revenue Analysis, Customer Segmentation, Customer Data, About Dataset
- Interactive Plotly charts with hover tooltips and colour-blind-friendly orange (churned) / teal (retained)
- Segment analysis by subscription, contract, payment method, gender and age group
- Searchable, sortable customer table with CSV download of the filtered rows
- Insights and recommendations generated from the data (nothing hard-coded), separating observed patterns, possible interpretations and actions
- Data validation (columns, missing values, duplicate IDs, negative values, unexpected Churn/Contract values) with a visible quality report; the dataset is never silently changed

## Technologies
Python 3.10+, Streamlit, Pandas, NumPy, Plotly, custom CSS.

## Dataset
`customer_churn_cleaned.csv` - 15 records, 11 columns: CustomerID, Gender, Age, TenureMonths, SubscriptionType, MonthlyCharges, TotalCharges, ContractType, SupportTickets, PaymentMethod, Churn. The CSV has no currency field; amounts are shown in INR as the project brief specifies.

## KPI definitions
| KPI | Formula |
|---|---|
| Churn rate | churned customers / total customers x 100 |
| Retention rate | retained customers / total customers x 100 |
| Monthly revenue at risk | sum of MonthlyCharges where Churn = Yes (not TotalCharges, not lifetime value) |
| Average tenure | mean of TenureMonths |

## Project structure
```
Customer-Churn-KPI-Dashboard/
├── app.py                    # Streamlit UI and navigation
├── utils.py                  # loading, validation, KPIs, insights
├── charts.py                 # Plotly figures
├── requirements.txt
├── README.md
├── customer_churn_cleaned.csv
├── assets/dashboard_logo.png
└── .gitignore
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. If the CSV is missing or renamed, the app looks for any CSV next to `app.py`, or you can upload one from the sidebar.

## How to test the filters
1. Contract type = Month-to-Month: 7 customers, churn rate 100%.
2. Tenure slider 0-10: only short-tenure customers remain; KPIs change.
3. Payment method = UPI with Month-to-Month: 3 customers, revenue at risk 149.97.
4. Combine all three, then press **Reset filters** to return to 15 customers.
5. Contract = Two Year with tenure 0-10 gives no customers and shows the "no match" message.
6. On Customer Data, search, sort and press the download button.

## Deploy to Streamlit Community Cloud
1. Create a **public** GitHub repository and push this folder:
   ```bash
   git init && git add . && git commit -m "Churn KPI dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Customer-Churn-KPI-Dashboard.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, branch `main` and main file `app.py`.
4. Click **Deploy**. When it finishes, copy the public `https://<name>.streamlit.app` URL and use it for submission and in this README.

## Limitations
- 15 records only: no statistical significance, and one customer can change a rate by about 7 points.
- Descriptive analysis: patterns are associations, not causes.
- In this sample every Month-to-Month customer churned, so some recommendations and the watchlist can be empty.
- Revenue at risk uses current monthly charges only.

## Screenshots
_Add screenshots of the Overview, Churn Analysis and Revenue pages here (e.g. `assets/overview.png`)._

## Conclusion
In this sample, churned customers have much shorter tenure, more support tickets and are all on Month-to-Month contracts. A larger dataset would be needed before acting on these patterns.
