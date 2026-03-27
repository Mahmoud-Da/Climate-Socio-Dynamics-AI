# 🌍 Climate-Socio-Dynamics AI

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview

This project builds a multivariate Artificial Intelligence model that integrates **historical weather data** (Temperature) with **anthropogenic socioeconomic data** (CO2 Emissions, GDP, Urbanization) to simulate climate trends over the next 50 years.

Going beyond simple univariate statistical predictions (like ARIMA), this project demonstrates how engineers can incorporate the complex, non-linear intertwining of multiple external factors into mathematical Deep Learning models using **Long Short-Term Memory (LSTM)** networks.

---

## 🏗️ Architecture & Engineering Phases

The project pipeline is strictly divided into three core engineering phases:

### Phase 1: Data Platform & Statistical Validation (`src/phase1_pipeline.py`)

- **Data Ingestion & Alignment:** Resolves frequency mismatches by upsampling yearly socioeconomic data (GDP, CO2) to match monthly weather data using cubic interpolation.
- **Correlation & Causation:** Proves that external variables legitimately influence temperature. We utilize the **Granger Causality Test** to statistically validate that historical CO2 trends possess predictive power over future temperature changes (p-value < 0.05).

### Phase 2: Multivariate Simulation Model (`src/phase2_model.py`)

- **Time-Series Deep Learning:** Implements a Multivariate LSTM in **PyTorch**. The network utilizes a sliding window (12-month sequence) to capture natural climate seasonality while factoring in external socioeconomic pressures.
- **Model Validation (Backtesting):** Validates the model's reproducibility by hiding the last 10 years of data from the training set. The model generates predictions for this hidden period, which are then compared against actual historical data to calculate the Root Mean Squared Error (RMSE).

### Phase 3: Scenario-Specific Simulation UI (`src/phase3_dashboard.py`)

- **Autoregressive Forecasting:** Predicts 50 years into the future by feeding the model's own predictions back into itself continuously.
- **Interactive Dashboard:** Utilizes **Streamlit** and **Plotly** to create a dynamic web interface. It allows stakeholders to inject different policy scenarios (_e.g., "Green Revolution", "Status Quo", "Accelerated Emissions"_) and visualize how human behavior alters the AI's 50-year climate trajectory.

---

## 📂 Project Structure

```text
├── data/
│   └── processed/
│       └── climate_socio_merged.csv       # Cleaned and merged dataset
├── models/
│   └── climate_lstm_v1.pth                # Saved PyTorch model weights
├── project-explain/
│   ├── 1_Introduction_to_ML_Process.txt   # Concept explanations
│   ├── 2_Phase1_Data_Analysis.txt
│   ├── 3_Phase2_The_AI_Model.txt
│   ├── 4_Phase3_Simulation_Dashboard.txt
│   ├── 5-quotation.txt                    # Q&A for presentations
│   └── INSTALL_AND_RUN.txt                # Absolute beginner setup guide
├── reports/
│   └── figures/                           # Generated analytical plots
│       ├── phase1_correlation_matrix.png
│       ├── phase1_trends.png
│       └── phase2_backtest.png
├── src/                                   # Core Python Scripts
│   ├── phase1_pipeline.py
│   ├── phase2_model.py
│   └── phase3_dashboard.py
├── Pipfile                                # Dependency management
├── Pipfile.lock
└── README.md
```

---

## 🚀 Quick Start / How to Run

_Note: For absolute beginners (or Mac users starting from scratch), please read `project-explain/INSTALL_AND_RUN.txt` for step-by-step system setup instructions._

### 1. Install Dependencies

This project uses `pipenv` for robust virtual environment management.

```bash
pipenv install
```

### 2. Execute the Pipeline

Run the phases in sequential order:

**Step A: Run Data Analysis (Phase 1)**

```bash
pipenv run python src/phase1_pipeline.py
```

_(Check `reports/figures/` for the generated correlation matrices and trend plots)._

**Step B: Train the AI Model (Phase 2)**

```bash
pipenv run python src/phase2_model.py
```

_(This will train the LSTM and save the weights to the `models/` directory)._

**Step C: Launch the Dashboard (Phase 3)**

```bash
pipenv run streamlit run src/phase3_dashboard.py
```

_(This opens the interactive web application in your default browser at `http://localhost:8501`)._

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Data Engineering:** Pandas, NumPy, Scikit-learn
- **Statistical Analysis:** Statsmodels (Granger Causality)
- **Deep Learning:** PyTorch (LSTM)
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Web Framework:** Streamlit
