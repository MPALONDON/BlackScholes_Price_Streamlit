# Black-Scholes P/L & Heatmap Explorer

A **Streamlit application** for calculating **Black-Scholes option prices**, simulating **Profit & Loss (P/L)** based on user purchase prices, visualizing **heatmaps of P/L**, and saving results to a **SQLite database** using **SQLAlchemy 2.0**.

---

## Features

- Calculate **Call and Put option prices** using the **Black-Scholes model**.
- Input **trade purchase prices** to calculate **P/L**.
- Generate **interactive P/L heatmaps** for varying **spot prices** and **volatility**.
- **Green/red heatmap** shows profitable vs loss scenarios.
- **Save all inputs and outputs** in a **SQLite database** for future reference.
- View **recent calculations and outputs** directly from the app.

---

## Requirements

- Streamlit  
- pandas  
- numpy  
- scipy  
- matplotlib  
- seaborn  
- SQLAlchemy  
