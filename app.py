import streamlit as st
import pandas as pd
import joblib

st.title("Advanced Sales Forecast System")

model = joblib.load("sales_model.pkl")

store = st.number_input("Store ID")
product = st.number_input("Product ID")
price = st.number_input("Price")
promotion = st.number_input("Promotion (0 or 1)")
stock = st.number_input("Stock Level")
day = st.number_input("Day")
month = st.number_input("Month")
year = st.number_input("Year")

if st.button("Predict Sales"):
    result = model.predict([[store,product,price,promotion,stock,day,month,year]])
    st.success(f"Predicted Revenue: {result[0]}")