import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Generate synthetic data
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 200, n_samples)
    price = size * 50 + np.random.normal(0, 5000, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# Train model
def train_model():
    df = generate_house_data()
    X = df[['size']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    return model, df

def main():
    st.title("🏠 House Price Prediction App")

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

        fig = px.scatter(
            df,
            x="size",
            y="price",
            title="House Size vs Price"
        )

        fig.add_scatter(
            x=[size],
            y=[predicted_price],
            mode="markers",
            marker=dict(size=15, color="red"),
            name="Prediction"
        )

        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
