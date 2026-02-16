import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os


class ClimateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ClimateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class ClimateSimulator:
    def __init__(self, model_path, data_path):
        self.device = torch.device('cpu')

        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.features = ['Temperature', 'CO2_Emissions', 'GDP', 'Urbanization']
        self.data_values = self.df[self.features].values

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.data_values)

        self.model = ClimateLSTM(input_size=4, hidden_size=64, output_size=1)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.eval()

    def generate_future_exogenous(self, months, scenario_type):
        """
        Generates future CO2, GDP, and Urbanization based on the chosen scenario.
        """
        last_vals = self.df.iloc[-1]

        future_co2 = []
        future_gdp = []
        future_urb = []

        current_co2 = last_vals['CO2_Emissions']
        current_gdp = last_vals['GDP']
        current_urb = last_vals['Urbanization']

        for i in range(months):

            if scenario_type == "Green Revolution (Successful Policy)":

                current_co2 -= 0.05
                current_gdp += 0.2
                current_urb += 0.01

            elif scenario_type == "Status Quo (Business as Usual)":

                current_co2 += 0.08
                current_gdp += 0.5
                current_urb += 0.02

            elif scenario_type == "Accelerated (High Emissions)":

                current_co2 += 0.25
                current_gdp += 0.8
                current_urb += 0.05

            noise = np.random.normal(0, 0.05)
            future_co2.append(current_co2 + noise)
            future_gdp.append(current_gdp + noise)
            future_urb.append(current_urb + noise)

        return np.array(future_co2), np.array(future_gdp), np.array(future_urb)

    def run_simulation(self, months=600, scenario="Status Quo (Business as Usual)"):
        """
        The Auto-Regressive Loop:
        Predict T+1 -> Use T+1 as input for T+2 -> Repeat
        """

        seq_length = 12
        history_scaled = self.scaler.transform(self.data_values[-seq_length:])
        current_seq = torch.tensor(
            history_scaled, dtype=torch.float32).unsqueeze(0)

        f_co2, f_gdp, f_urb = self.generate_future_exogenous(months, scenario)

        future_temps = []

        min_vals = self.scaler.data_min_
        range_vals = self.scaler.data_range_

        for i in range(months):
            with torch.no_grad():

                pred_temp_scaled = self.model(current_seq).item()

            real_temp = (pred_temp_scaled * range_vals[0]) + min_vals[0]
            future_temps.append(real_temp)

            next_co2_scaled = (f_co2[i] - min_vals[1]) / range_vals[1]
            next_gdp_scaled = (f_gdp[i] - min_vals[2]) / range_vals[2]
            next_urb_scaled = (f_urb[i] - min_vals[3]) / range_vals[3]

            new_row = torch.tensor(
                [[pred_temp_scaled, next_co2_scaled, next_gdp_scaled, next_urb_scaled]], dtype=torch.float32)

            current_seq = torch.cat(
                (current_seq[:, 1:, :], new_row.unsqueeze(0)), dim=1)

        return future_temps, f_co2


def main():
    st.set_page_config(page_title="AI Climate 50-Year Sim", layout="wide")

    st.title("🌍 50-Year Climate AI Simulation")
    st.markdown("""
    **Project Goal:** Integrate historical weather data with socioeconomic factors to simulate future climate trends using LSTM Neural Networks.
    """)

    st.sidebar.header("Scenario Settings")
    scenario = st.sidebar.selectbox(
        "Select Future Scenario:",
        ("Green Revolution (Successful Policy)",
         "Status Quo (Business as Usual)",
         "Accelerated (High Emissions)")
    )

    years = st.sidebar.slider("Simulation Duration (Years)", 10, 50, 50)
    months = years * 12

    if st.sidebar.button("Run Simulation"):
        with st.spinner('Running AI Inference...'):
            try:
                sim = ClimateSimulator(
                    'models/climate_lstm_v1.pth', 'data/processed/climate_socio_merged.csv')
                preds, co2_vals = sim.run_simulation(months, scenario)

                last_date = pd.to_datetime('2024-12-31')
                future_dates = pd.date_range(
                    start=last_date, periods=months, freq='ME')

                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=future_dates, y=preds, mode='lines', name='Predicted Temp (°C)', line=dict(color='red')))
                fig_temp.update_layout(
                    title=f"Projected Temperature Trend ({scenario})", xaxis_title="Year", yaxis_title="Temperature (°C)")

                fig_co2 = go.Figure()
                fig_co2.add_trace(go.Scatter(x=future_dates, y=co2_vals, mode='lines',
                                  name='Projected CO2', line=dict(color='green', dash='dot')))
                fig_co2.update_layout(
                    title="Underlying CO2 Assumption (ppm)", xaxis_title="Year", yaxis_title="CO2 Emissions")

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_temp, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_co2, use_container_width=True)

                st.subheader("Simulation Analysis")
                delta = preds[-1] - preds[0]
                st.metric(label="Total Temperature Change",
                          value=f"{delta:.2f} °C", delta=f"{delta:.2f} °C", delta_color="inverse")

                st.info("Note: The 'Seasonality' (waves) you see is learned by the AI from historical data. The 'Trend' (up/down) is driven by the CO2/GDP scenario inputs.")

            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.warning(
                    "Did you run Phase 2 to generate 'models/climate_lstm_v1.pth'?")

    else:
        st.info("👈 Select a scenario and click 'Run Simulation' to start.")


if __name__ == "__main__":
    main()
