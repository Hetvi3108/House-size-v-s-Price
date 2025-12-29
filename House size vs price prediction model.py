import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic house data
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 200, n_samples)
    price = size * 50 + np.random.normal(0, 5000, n_samples)
    return pd.DataFrame({"size": size, "price": price})

# Train ML model
def train_model():
    df = generate_house_data()
    X = df[["size"]]
    y = df["price"]

    model = LinearRegression()
    model.fit(X, y)
    return model, df

def main():
    st.title("🏠 House Size vs Price Prediction")

    model, df = train_model()

    size = st.number_input(
        "Enter House Size (sq ft)",
        min_value=500,
        max_value=3000,
        value=1500
    )

    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])[0]

        st.success(f"Estimated House Price: ₹ {predicted_price:,.2f}")

        # Plot using matplotlib
        fig, ax = plt.subplots()
        ax.scatter(df["size"], df["price"], label="Training Data")
        ax.scatter(size, predicted_price, color="red", s=100, label="Prediction")
        ax.set_xlabel("House Size (sq ft)")
        ax.set_ylabel("Price")
        ax.set_title("House Size vs Price")
        ax.legend()

        st.pyplot(fig)

if __name__ == "__main__":
    main()
