import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Sample data 
np.random.seed(0)
data = {
    'CustomerID': np.random.choice(range(1, 101), size=500),
    'Date': pd.date_range(start='2023-01-01', periods=500, freq='D'),
    'Amount': np.random.randint(10, 200, size=500)
}

# DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())


# Calculate RFM metrics
current_date = df['Date'].max()
rfm = df.groupby('CustomerID').agg({
    'Date': ['max', 'count'],  # Calculate max date for recency and count for frequency
    'Amount': 'sum'             # Calculate total amount for monetary
})

# Calculate recency in days
rfm['Recency'] = (current_date - rfm[('Date', 'max')]).dt.days

# Flatten the MultiIndex columns
rfm.columns = ['_'.join(col).rstrip('_') if isinstance(col, tuple) else col.rstrip('_') for col in rfm.columns]

# Rename columns
rfm.rename(columns={'Date_max': 'LastPurchase', 'Date_count': 'Frequency', 'Amount_sum': 'Monetary'}, inplace=True)

# Reset index
rfm = rfm.reset_index()

# Display the first few rows of the RFM DataFrame
print(rfm.head())


# Assign RFM scores
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Calculate overall RFM score
rfm['RFM_Score'] = rfm[['R_score', 'F_score', 'M_score']].sum(axis=1)

# Display the first few rows of the RFM DataFrame with scores
print(rfm.head())


# Define segments
def classify_segment(rfm_score):
    if rfm_score >= 13:
        return 'Champions'
    elif rfm_score >= 9:
        return 'Potential Loyalists'
    elif rfm_score >= 5:
        return 'At Risk'
    else:
        return 'Hibernating'

# Classify customers
rfm['Segment'] = rfm['RFM_Score'].apply(classify_segment)

# Display the first few rows of the RFM DataFrame with segments
print(rfm.head())



# Plot the distribution of segments
segment_counts = rfm['Segment'].value_counts()

plt.figure(figsize=(10, 6))
segment_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon', 'lightcoral'])
plt.title('Customer Segments Distribution')
plt.xlabel('Segments')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()




