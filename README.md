# Retail Chain Forecasting and Customer Analytics System

This project provides a complete, end-to-end analytics platform for retail decision-making. It integrates product-level forecasting, customer segmentation, and actionable insights into a unified Python-based workflow.

## Overview
The system helps retailers make informed decisions related to demand forecasting, inventory planning, customer retention, and product strategy. It combines machine learning, time-series modeling, and RFM-based CRM analytics to deliver reliable forecasts and meaningful insights.

## Key Features

### 1. Automated Data Cleaning and Preprocessing
- Removes negative or invalid values
- Standardizes column names
- Fixes missing data
- Computes TotalPrice automatically
- Applies weekly aggregation for forecasting stability

### 2. Product-Level Sales Forecasting
The system supports two forecasting models:
- **SARIMA** for long-term, trend-aware forecasting
- **XGBoost** for short-term, high-accuracy predictions

Both forecasts are displayed on the same graph when a product is entered, supporting manufacturing and inventory planning.

### 3. Customer Segmentation (RFM Analysis)
Customers are categorized using Recency, Frequency, and Monetary scoring:
- Top Spenders
- New Customers
- At-Risk / Dormant

Each group automatically receives tailored discount recommendations:
- 15% VIP Discount (Top Spenders)  
- 5% Welcome Offer (New Customers)  
- 10% Winback Coupon (At-Risk)

### 4. Customer Lookup Module
Retailers can search any Customer ID to view:
- Complete RFM profile
- Purchase history
- Total spend and quantity
- Most purchased items

### 5. Product Performance Insights
The system identifies:
- Top 20 best-selling products
- Bottom 20 worst-performing products

Metrics include total sales, quantity sold, and unique customers.

## Technologies Used
- Python (pandas, numpy, matplotlib)
- Statsmodels (SARIMA)
- XGBoost
- Scikit-learn
- OpenPyXL
- IPython Display

## How to Use

### 1. Forecast Sales
Enter a product name or stock code to generate forecasts.

### 2. Customer Insights
Enter a Customer ID to view their RFM profile and purchase history.

## Outcome
This system enables retail businesses to understand customer behavior, anticipate product demand, and optimize stocking and marketing decisions through a unified predictive analytics platform.

