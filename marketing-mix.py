import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Synthetic Dataset
np.random.seed(42)
n_weeks = 104
weeks = np.arange(1, n_weeks + 1)

# Ad spend with some random fluctuations
tv_spend_np = 5 + 2 * np.sin(weeks / 10) + np.random.normal(0, 1, n_weeks)
tv_spend = pd.Series(tv_spend_np) # Convert NumPy array to Pandas Series
digital_spend = 3 + 1.5 * np.cos(weeks / 12) + np.random.normal(0, 0.8, n_weeks)
radio_spend = 1 + 0.5 * np.sin(weeks / 8 + np.pi/4) + np.random.normal(0, 0.3, n_weeks)
search_impressions = 100 + 10 * tv_spend_np + np.random.normal(0, 15, n_weeks) # Using tv_spend_np here

# Seasonality
seasonality = 20 * np.sin(2 * np.pi * weeks / 52)

# Base sales with adstock effect (simplified) and seasonality
base_sales = 100 + seasonality
sales = base_sales + 0.8 * tv_spend.shift(1, fill_value=tv_spend[0]) + 0.5 * digital_spend + 0.2 * radio_spend + 0.05 * search_impressions + np.random.normal(0, 10, n_weeks)

# Create Pandas DataFrame
data = pd.DataFrame({
    'week': weeks,
    'sales': sales,
    'tv_spend': tv_spend_np, # Using the NumPy array here to match the DataFrame structure
    'digital_spend': digital_spend,
    'radio_spend': radio_spend,
    'search_impressions': search_impressions,
    'seasonality': seasonality
})

# Display the first few rows of the dataset
print("Sample of the Synthetic Dataset:")
print(data.head())
print("\n")

# 2. Fit a Regression Model
X = data[['tv_spend', 'digital_spend', 'radio_spend', 'search_impressions', 'seasonality']]
y = data['sales']
X = sm.add_constant(X)  # Add a constant (intercept) to the model

model = sm.OLS(y, X).fit()

# Print the model summary
print("Regression Model Summary:")
print(model.summary())
print("\n")

# 3. Interpret Coefficients
print("Coefficients Interpretation:")
print(f"Intercept: {model.params['const']:.2f} (Base sales when all predictors are zero)")
print(f"TV Spend Coefficient: {model.params['tv_spend']:.2f} (For every $1k increase in TV spend, sales increase by approximately ${model.params['tv_spend']:.2f}, holding other factors constant)")
print(f"Digital Spend Coefficient: {model.params['digital_spend']:.2f} (For every $1k increase in digital spend, sales increase by approximately ${model.params['digital_spend']:.2f}, holding other factors constant)")
print(f"Radio Spend Coefficient: {model.params['radio_spend']:.2f} (For every $1k increase in radio spend, sales increase by approximately ${model.params['radio_spend']:.2f}, holding other factors constant)")
print(f"Search Impressions Coefficient: {model.params['search_impressions']:.4f} (For every 1 unit increase in search impressions, sales increase by approximately ${model.params['search_impressions']:.4f}, holding other factors constant)")
print(f"Seasonality Coefficient: {model.params['seasonality']:.2f} (The impact of the seasonality factor on sales)")
print("\n")

# 4. Visualize Model Fit and Channel Contributions

# Model Fit Visualization
plt.figure(figsize=(12, 6))
plt.plot(data['week'], data['sales'], label='Actual Sales')
plt.plot(data['week'], model.fittedvalues, label='Predicted Sales', color='red')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.grid(True)
plt.show()

# Channel Contributions Visualization
contributions = model.params[['tv_spend', 'digital_spend', 'radio_spend', 'search_impressions']] * data[['tv_spend', 'digital_spend', 'radio_spend', 'search_impressions']].mean()

plt.figure(figsize=(10, 6))
contributions.sort_values(ascending=True).plot(kind='barh', color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
plt.xlabel('Average Sales Contribution')
plt.ylabel('Marketing Channel')
plt.title('Average Sales Contribution by Marketing Channel')
plt.grid(axis='x')
plt.show()

# Decomposition of Sales
base = model.params['const'] + model.params['seasonality'] * data['seasonality']
tv_contribution = model.params['tv_spend'] * data['tv_spend']
digital_contribution = model.params['digital_spend'] * data['digital_spend']
radio_contribution = model.params['radio_spend'] * data['radio_spend']
search_contribution = model.params['search_impressions'] * data['search_impressions']
residual = model.resid

plt.figure(figsize=(14, 8))
plt.plot(data['week'], data['sales'], label='Actual Sales', color='black')
plt.plot(data['week'], base, label='Base + Seasonality', linestyle='--')
plt.plot(data['week'], base + tv_contribution, label='Base + Seasonality + TV', linestyle='--')
plt.plot(data['week'], base + tv_contribution + digital_contribution, label='Base + Seasonality + TV + Digital', linestyle='--')
plt.plot(data['week'], base + tv_contribution + digital_contribution + radio_contribution, label='Base + Seasonality + TV + Digital + Radio', linestyle='--')
plt.plot(data['week'], model.fittedvalues, label='Predicted Sales (Full Model)', linestyle='-', color='red')
plt.xlabel('Week')
plt.ylabel('Sales')
plt.title('Decomposition of Sales by Contributing Factors')
plt.legend()
plt.grid(True)
plt.show()