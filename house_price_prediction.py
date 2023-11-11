import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
data = pd.read_csv('houseprice.csv')

# Separate features and target variable
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1500)

# Train the model
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Streamlit app
st.title("House Price Prediction App")

# Add sliders for user input
living_area = st.slider("Living Area", min_value=1.0, max_value=3.0, step=0.001, value=1.982)
bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=2.0, step=0.1, value=1.0)
bedrooms = st.slider("Bedrooms", min_value=1, max_value=5, step=1, value=3)
lot_size = st.slider("Lot Size", min_value=0.1, max_value=1.0, step=0.01, value=0.38)
age = st.slider("Age", min_value=1, max_value=150, step=1, value=14)
fireplace = st.slider("Fireplace", min_value=0, max_value=1, step=1, value=1)

# Make predictions based on user input
user_input = [[living_area, bathrooms, bedrooms, lot_size, age, fireplace]]
prediction = model.predict(user_input)

# Display the prediction
st.subheader("Prediction:")
st.write(f"The predicted house price is: ${round(prediction[0], 4)}")
