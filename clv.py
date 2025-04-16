import pandas as pd

# Sample dataset for individual customers
data = {
    'CustomerID': [1, 2, 3],
    'AverageOrderValue': [100, 200, 50],
    'PurchaseFrequency': [12, 6, 24],  # Purchases per year
    'CustomerLifespan': [5, 3, 2],  # Years
    'Margin': [0.2, 0.3, 0.1]  # Margin as a proportion of order value
}
# Discount rate for CLV calculation
discount_rate = 0.1  # 10%

# DataFrame
df = pd.DataFrame(data)



# Function to calculate individual Customer Lifetime Value (CLV)
def calculate_individual_clv(row: pd.Series) -> float:
    """
    Calculates the Customer Lifetime Value (CLV) for an individual customer.

    CLV is calculated using the discounted cash flow method over the customer's lifespan.

    Args:
        row (pd.Series): A Pandas Series representing a customer's data,
                          including 'AverageOrderValue', 'Margin', 'PurchaseFrequency',
                          and 'CustomerLifespan'.

    Returns:
        float: The calculated Customer Lifetime Value (CLV) for the customer.
    """
    clv: float = 0.0
    # Ensure CustomerLifespan is treated as an integer for loop range
    lifespan: int = int(row['CustomerLifespan'])
    for t in range(1, lifespan + 1):
        clv += (row['AverageOrderValue'] * row['Margin'] * row['PurchaseFrequency']) / ((1 + discount_rate) ** t)
    return clv


# Apply the function to each customer
df['CLV'] = df.apply(calculate_individual_clv, axis=1)

# Display the DataFrame with CLV
print(df[['CustomerID', 'CLV']])
