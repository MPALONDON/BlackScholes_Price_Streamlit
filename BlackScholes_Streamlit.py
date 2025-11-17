import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine,Integer,Numeric,Boolean,ForeignKey,DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, mapped_column, Mapped
from datetime import datetime


class BlackScholes:
    def __init__(self, time_to_maturity: float, strike:float, current_price:float, volatility:float, interest_rate:float):
        self.time_to_maturity = float(time_to_maturity)
        self.strike = float(strike)
        self.current_price = float(current_price)
        self.volatility = float(volatility)
        self.interest_rate = float(interest_rate)

    def calculate_prices(self):
        T = max(self.time_to_maturity, 1e-12)
        vol = max(self.volatility, 1e-12)

        d1 = (
            log(self.current_price / self.strike)
            + (self.interest_rate + 0.5 * vol ** 2) * T
        ) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)

        call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-self.interest_rate * T) * norm.cdf(d2)
        )
        put_price = (
            self.strike * exp(-self.interest_rate * T) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1

        call_gamma = norm.pdf(d1) / (self.current_price * vol * sqrt(T))


        self.call_price = float(call_price)
        self.put_price = float(put_price)
        self.call_delta = float(call_delta)
        self.put_delta = float(put_delta)
        self.call_gamma = float(call_gamma)
        self.put_gamma = float(call_gamma)

        return self.call_price, self.put_price

os.makedirs("instance", exist_ok=True)
DB_URI = "sqlite:///instance/bello_pluto.db"

engine = create_engine(DB_URI, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)
Base = declarative_base()

class BlackScholesInput(Base):
    __tablename__ = "BlackScholesInputs"

    CalculationId:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    StockPrice:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    StrikePrice:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    InterestRate:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    Volatility:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    TimeToExpiry:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    CallPurchasePrice:Mapped[float] = mapped_column(Numeric(18,9), nullable=True)
    PutPurchasePrice:Mapped[float] = mapped_column(Numeric(18,9), nullable=True)
    CreatedAt:Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class BlackScholesOutput(Base):
    __tablename__ = "BlackScholesOutputs"

    CalculationOutputId:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    VolatilityShock:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    StockPriceShock:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    OptionPrice:Mapped[float] = mapped_column(Numeric(18, 9), nullable=False)
    IsCall:Mapped[bool] = mapped_column(Boolean, nullable=False)
    CalculationId:Mapped[int] = mapped_column(Integer, ForeignKey("BlackScholesInputs.CalculationId"), nullable=False)


Base.metadata.create_all(bind=engine)

st.set_page_config(page_title="Black-Scholes P/L", layout="wide", page_icon="📈")

st.title("Black-Scholes — Option P/L, Heatmaps")


with st.sidebar:
    st.header("Model inputs")
    current_price = st.number_input("Current Asset Price", value=100.0, min_value=0.0001, format="%.2f")
    strike = st.number_input("Strike Price", value=100.0, min_value=0.0001, format="%.2f")
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.0, format="%.2f")
    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, max_value=10.0, format="%.2f")
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, format="%.2f")
    st.markdown("---")
    st.header("Trade inputs")
    call_purchase_price = st.number_input("Call Purchase Price", value=5.0, min_value=0.0, format="%.2f")
    put_purchase_price = st.number_input("Put Purchase Price", value=5.0, min_value=0.0, format="%.2f")

    st.markdown("---")
    st.header("Heatmap setup")
    spot_min = st.number_input("Min Spot Price", value=current_price * 0.8, min_value=0.0001, format="%.2f")
    spot_max = st.number_input("Max Spot Price", value=current_price * 1.2, min_value=0.0001, format="%.2f")
    vol_min = st.slider("Min Volatility for Heatmap", min_value=0.01, max_value=1.0, value=max(0.01, volatility * 0.5), step=0.01)
    vol_max = st.slider("Max Volatility for Heatmap", min_value=0.01, max_value=1.0, value=min(1.0, volatility * 1.5), step=0.01)
    grid_size = st.slider("Heatmap resolution (n x n)", 5, 25, 10)


input_df = pd.DataFrame({
    "Current Asset Price":[current_price],
    "Strike Price":[strike],
    "Time to Maturity (Years)":[time_to_maturity],
    "Volatility (σ)":[volatility],
    "Risk-Free Rate":[interest_rate],
    "Call Purchase Price":[call_purchase_price],
    "Put Purchase Price":[put_purchase_price]
})
st.subheader("Inputs")
st.table(input_df)


bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()


call_pl = call_price - float(call_purchase_price)
put_pl = put_price - float(put_purchase_price)


def colored_pl(value):
    color = "green" if value >= 0 else "red"
    sign = "+" if value >= 0 else ""
    return f"<span style='color:{color}; font-weight:bold;'>{sign}£{value:,.2f}</span>"


col1, col2 = st.columns(2)
with col1:
    st.markdown("### Theoretical Option Prices")
    st.write(f"Call Price: £{call_price:,.2f}")
    st.write(f"Put Price:  £{put_price:,.2f}")

with col2:
    st.markdown("### Profit / Loss (P/L) given purchase price")
    st.markdown(f"Call P/L: {colored_pl(call_pl)}", unsafe_allow_html=True)
    st.markdown(f"Put P/L: {colored_pl(put_pl)}", unsafe_allow_html=True)

st.markdown("---")

if st.button("Clear Database"):
    session = SessionLocal()
    session.query(BlackScholesInput).delete(synchronize_session=False)
    session.query(BlackScholesOutput).delete(synchronize_session=False)
    session.commit()
    st.success("Database has been cleared")

if st.button("Calculate & Save (inputs + heatmap outputs)"):

    session = SessionLocal()
    try:
        input_row = BlackScholesInput(
            StockPrice=current_price,
            StrikePrice=strike,
            InterestRate=interest_rate,
            Volatility=volatility,
            TimeToExpiry=time_to_maturity,
            CallPurchasePrice=call_purchase_price,
            PutPurchasePrice=put_purchase_price
        )
        session.add(input_row)
        session.commit()
        session.refresh(input_row)
        calc_id = int(input_row.CalculationId)


        spot_range = np.linspace(spot_min, spot_max, grid_size)
        vol_range = np.linspace(vol_min, vol_max, grid_size)


        outputs_to_add = []

        for vol in vol_range:
            for spot in spot_range:
                bs_temp = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate)
                call_p_temp, put_p_temp = bs_temp.calculate_prices()

                out_call = BlackScholesOutput(
                    VolatilityShock=vol,
                    StockPriceShock=spot,
                    OptionPrice=float(call_p_temp),
                    IsCall=True,
                    CalculationId=calc_id
                )
                outputs_to_add.append(out_call)

                out_put = BlackScholesOutput(
                    VolatilityShock=vol,
                    StockPriceShock=spot,
                    OptionPrice=float(put_p_temp),
                    IsCall=False,
                    CalculationId=calc_id
                )
                outputs_to_add.append(out_put)

        session.add_all(outputs_to_add)
        session.commit()
        st.success(f"Saved inputs and {len(outputs_to_add)} output rows (calc id {calc_id}) to DB.")
    except Exception as e:
        session.rollback()
        st.error(f"Error saving to DB: {e}")
    finally:
        session.close()


spot_range = np.linspace(spot_min, spot_max, grid_size)
vol_range = np.linspace(vol_min, vol_max, grid_size)

call_pl_matrix = np.zeros((len(vol_range), len(spot_range)))
put_pl_matrix = np.zeros((len(vol_range), len(spot_range)))

for i, vol in enumerate(vol_range):
    for j, spot in enumerate(spot_range):
        bs_temp = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate)
        c_price, p_price = bs_temp.calculate_prices()

        call_pl_matrix[i, j] = float(c_price) - float(call_purchase_price)
        put_pl_matrix[i, j] = float(p_price) - float(put_purchase_price)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(call_pl_matrix, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
            annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax1)
ax1.set_title("Call P/L Heatmap (green = profit)")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(put_pl_matrix, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
            annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax2)
ax2.set_title("Put P/L Heatmap (green = profit)")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility")
st.pyplot(fig2)

st.markdown("---")

with st.expander("Show recent saved calculations (inputs)"):
    session = SessionLocal()
    try:
        results = (
            session.query(BlackScholesInput)
            .order_by(BlackScholesInput.CalculationId.desc())
            .limit(20)
            .all()
        )

        if results:
            df = pd.DataFrame([
                {
                    "CalculationId": r.CalculationId,
                    "StockPrice": float(r.StockPrice),
                    "StrikePrice": float(r.StrikePrice),
                    "InterestRate": float(r.InterestRate),
                    "Volatility": float(r.Volatility),
                    "TimeToExpiry": float(r.TimeToExpiry),
                    "CallPurchasePrice": float(r.CallPurchasePrice),
                    "PutPurchasePrice": float(r.PutPurchasePrice),
                    "CreatedAt": r.CreatedAt,
                }
                for r in results
            ])
            st.dataframe(df)
        else:
            st.write("No saved calculations yet.")

    except Exception as e:
        st.error(f"DB read error: {e}")
    finally:
        session.close()

    database_table = pd.read_sql_table("BlackScholesInputs", engine)
    if len(database_table) >0:
        df = database_table.to_csv(index=False)
        st.download_button(
            "Download Inputs",
            df,
            "Inputs.csv",
            "text/csv"
        )

with st.expander("Show sample saved outputs (most recent 50 rows)"):
    session = SessionLocal()
    try:
        results = (
            session.query(BlackScholesOutput)
            .order_by(BlackScholesOutput.CalculationOutputId.desc())
            .limit(50)
            .all()
        )

        if results:
            df = pd.DataFrame([
                {
                    "CalculationOutputId": r.CalculationOutputId,
                    "VolatilityShock": float(r.VolatilityShock),
                    "StockPriceShock": float(r.StockPriceShock),
                    "OptionPrice": float(r.OptionPrice),
                    "IsCall": r.IsCall,
                    "CalculationId": r.CalculationId,
                }
                for r in results
            ])
            st.dataframe(df)
        else:
            st.write("No saved outputs yet.")

    except Exception as e:
        st.error(f"DB read error: {e}")
    finally:
        session.close()

    database_table = pd.read_sql_table("BlackScholesOutputs", engine)
    if len(database_table) > 0:
        df = database_table.to_csv(index=False)
        st.download_button(
            "Download Outputs",
            df,
            "Outputs.csv",
            "text/csv"
        )

st.caption("Heatmaps show P/L. Green = profit, Red = loss.")
