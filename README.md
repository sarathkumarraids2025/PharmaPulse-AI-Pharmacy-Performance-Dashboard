# 💊 PharmaPulse AI – Pharmacy Performance Dashboard

## 📌 Project Overview

**PharmaPulse AI** is an interactive pharmacy performance analytics dashboard designed to analyze pharmacy sales, product performance, profitability, and inventory insights using data analytics.

The project transforms raw pharmacy sales data into meaningful business insights through interactive visualizations, key performance indicators (KPIs), and data-driven reports.

This project was developed as part of the **EdVyro Data Analytics Program – Task 3**.

## 🎯 Objectives

- Analyze pharmacy sales performance and trends.
- Identify top-performing products and categories.
- Calculate revenue, profit, and profit margins.
- Compare branded and generic product performance.
- Analyze product contribution to overall revenue.
- Identify stock and inventory priorities where data is available.
- Generate actionable insights for pharmacy business decision-making.

## 📊 Key Features

- Interactive dashboard with dynamic filters.
- Total Revenue and Sales KPIs.
- Total Quantity Sold.
- Total Profit and Profit Margin.
- Sales trend analysis.
- Product performance analysis.
- Category-wise sales comparison.
- Branded vs Generic product analysis.
- Top-performing and low-performing products.
- Inventory analysis (when supported by dataset).
- Data cleaning and validation.
- Automated performance summary report.
- Exportable analytical outputs.

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Streamlit**
- **Plotly**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

## 📂 Dataset

The project uses the pharmacy sales sample dataset provided by EdVyro.

**Dataset:**  
https://edvyro.in/datasets/pharmacy_sales_sample.csv

The dataset is analyzed for:

- Sales transactions
- Product information
- Product categories
- Quantity sold
- Revenue
- Cost
- Profitability
- Sales dates
- Brand classification
- Inventory-related information (if available)

All analytical results are generated from the actual dataset. No artificial KPI values or fabricated business results are used.

## 📈 Dashboard Modules

### 1. Overview Dashboard
Provides a high-level summary of pharmacy business performance.

- Total Revenue
- Total Sales
- Total Quantity Sold
- Total Profit
- Profit Margin
- Overall performance trends

### 2. Sales Trend Analysis
Analyzes sales performance over time.

- Daily sales trends
- Monthly sales trends
- Revenue growth
- Sales patterns

### 3. Product Performance Analysis
Identifies products contributing the most to pharmacy revenue.

- Top-selling products
- Revenue contribution
- Quantity sold
- Product profitability
- Low-performing products

### 4. Category Analysis
Evaluates pharmacy product categories.

- Category-wise revenue
- Category-wise sales volume
- Profitability comparison
- Category contribution

### 5. Branded vs Generic Analysis
Compares branded and generic products based on available dataset information.

- Revenue comparison
- Quantity sold
- Product contribution
- Profitability comparison

### 6. Profitability Analysis
Analyzes financial performance.

- Revenue
- Cost
- Profit
- Profit margin
- Product-level profitability

### 7. Inventory Insights
Analyzes stock-related information if the dataset contains inventory fields.

- Stock availability
- Stock priority
- Low-stock products
- Inventory-related insights

## 🧹 Data Cleaning & Validation

The project includes data preparation and validation procedures:

- Missing value analysis
- Duplicate record detection
- Data type validation
- Date validation
- Sales and quantity validation
- Cost and revenue validation
- Invalid value detection
- Data consistency checks

Cleaning rules are documented and applied reproducibly.

## 📁 Project Structure

```text
PharmaPulse-AI-Pharmacy-Performance-Dashboard/
│
├── app.py
├── data_loader.py
├── data_cleaning.py
├── analytics.py
├── inventory_analysis.py
├── report_generator.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── pharmacy_sales_sample.csv
│
├── notebooks/
│   └── pharmacy_sales_analysis.ipynb
│
├── outputs/
│   ├── cleaned_pharmacy_sales.csv
│   ├── product_performance.csv
│   ├── category_performance.csv
│   ├── brand_generic_comparison.csv
│   └── pharmacy_performance_report.md
│
└── tests/
    ├── test_data_cleaning.py
    ├── test_analytics.py
    └── test_inventory_analysis.py
```

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sarathkumarraids2025/PharmaPulse-AI-Pharmacy-Performance-Dashboard.git
```

### Step 2: Navigate to the Project Directory

```bash
cd PharmaPulse-AI-Pharmacy-Performance-Dashboard
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## 📋 Deliverables

- Interactive Pharmacy Performance Dashboard
- Cleaned pharmacy sales dataset
- Data analysis notebook
- Product performance analysis
- Category performance analysis
- Branded vs Generic comparison
- Pharmacy performance summary report
- Data validation and cleaning scripts

## 📌 Business Insights

The dashboard helps pharmacy business stakeholders:

- Understand sales performance.
- Identify revenue-generating products.
- Compare product categories.
- Evaluate product profitability.
- Understand branded and generic sales patterns.
- Identify potential inventory priorities.
- Make data-driven business decisions.

## ⚠️ Disclaimer

This project is developed for educational and data analytics purposes as part of the EdVyro Data Analytics Program.

The dashboard focuses on business performance analytics and does not provide medical advice, clinical recommendations, or treatment-related decisions.

## 👨‍💻 Author

**Sarathkumar R**

B.Tech – Artificial Intelligence and Data Science  
Chennai Institute of Technology

GitHub: [sarathkumarraids2025](https://github.com/sarathkumarraids2025)

---

⭐ If you find this project useful, consider giving the repository a star.
