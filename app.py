import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from joblib import load
import pandas as pd

# Load the trained model
model = load(open('ccf.joblib','rb'))

# Function to predict fraud
def predict_fraud(distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order):
    # Prepare the input data in the same format as the training data
    input_data = [[
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        repeat_retailer,
        used_chip,
        used_pin_number,
        online_order
    ]]
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    return prediction[0]

# Streamlit application
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Credit Card Fraud Detection Dashboard")

# Main panel for user inputs and results
st.subheader("Enter Transaction Details")
col1, col2 = st.columns(2)

with col1:
    st.write("Distance from Home (Kilometer):")
    distance_from_home = st.number_input("Distance from Home (miles)", min_value=0.0, format="%.2f", key="home_dist", step=0.1, label_visibility="collapsed")
    
    st.write(distance_from_home)

with col2:
    st.write("Distance from Last Transaction (Kilometer):")
    distance_from_last_transaction = st.number_input("Distance from Last Transaction (miles)", min_value=0.0, format="%.2f", key="last_trans_dist", step=0.1, label_visibility="collapsed")
    st.write(distance_from_last_transaction)



col3, col4 = st.columns(2)
with col3:
    st.write("Purchased Price of Transaction (₹):")
    purchased_price_transaction = st.number_input("Purchased Price of Transaction ($)", min_value=0.0, format="%.2f", key="purchase_price", step=0.1, label_visibility="collapsed")
    st.write(purchased_price_transaction)
    
with col4:
    st.write("Median Purchase Price (₹):")
    median_purchase_price = st.number_input("Median Purchase Price (₹)", min_value=0.0, format="%.2f", key="median_price", step=0.1, label_visibility="collapsed")
    st.write(median_purchase_price)

ratio_to_median_purchase_price = purchased_price_transaction / median_purchase_price if median_purchase_price != 0 else 0


col5, col6, col7, col8 = st.columns(4)
with col5:
    repeat_retailer = st.checkbox("Repeat Retailer", value=False)
    st.write("Yes" if repeat_retailer else "No")


with col6:
    used_chip = st.checkbox("Used Chip", value=False)
    st.write("Yes" if used_chip else "No")

with col7:
    used_pin_number = st.checkbox("Used Pin Number", value=False)
    st.write("Yes" if used_pin_number else "No")

with col8:
    online_order = st.checkbox("Online Order", value=False)
    st.write("Yes" if online_order else "No")

# Convert checkbox inputs to float (1.0 if checked, 0.0 if not)
repeat_retailer = 1.0 if repeat_retailer else 0.0
used_chip = 1.0 if used_chip else 0.0
used_pin_number = 1.0 if used_pin_number else 0.0
online_order = 1.0 if online_order else 0.0

# Predict fraud when the button is clicked
if st.button("Predict Fraud", use_container_width=True):
    if distance_from_home == 0.0 or distance_from_last_transaction == 0.0 or purchased_price_transaction == 0.0 or median_purchase_price == 0.0:
        st.error("Please Enter the details First")
    else:
        prediction = predict_fraud(
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_retailer,
            used_chip,
            used_pin_number,
            online_order
        )
        
        result = "Fraud" if prediction == 1.0 else "Not Fraud"
        st.success(f"Prediction: {result}")

# Display input summary in a larger and clearer format
st.subheader("Summary")
input_summary = {
    "Distance from Home (Kilometers)": distance_from_home,
    "Distance from Last Transaction (Kilometers)": distance_from_last_transaction,
    "Purchased Price of Transaction (₹)": purchased_price_transaction,
    "Median Purchase Price (₹)": median_purchase_price,
    "Ratio to Median Purchase Price": ratio_to_median_purchase_price,
    "Repeat Retailer": "Yes" if repeat_retailer else "No",
    "Used Chip": "Yes" if used_chip else "No",
    "Used Pin Number": "Yes" if used_pin_number else "No",
    "Online Order": "Yes" if online_order else "No"
}
input_summary_df = pd.DataFrame([input_summary])
#st.table(input_summaray_df)
st.markdown(input_summaray_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)


