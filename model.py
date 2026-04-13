import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv("data.csv")

# Convert date
data['date'] = pd.to_datetime(data['date'], dayfirst=True)

# Extract date features
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

#  Convert text columns to numbers
data['promotion'] = data['promotion'].astype('category').cat.codes
data['product_category'] = data['product_category'].astype('category').cat.codes
data['customer_segment'] = data['customer_segment'].astype('category').cat.codes

# Features
X = data[['store_id','product_id','price','promotion','stock_level','day','month','year']]

# Target
y = data['revenue']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "sales_model.pkl")

print("Model trained successfully!")