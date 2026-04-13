import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

# Convert date column
data['date'] = pd.to_datetime(data['date'], dayfirst=True)

# Group sales by date
sales = data.groupby('date')['revenue'].sum()

# Plot graph
sales.plot()

plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")

plt.show()