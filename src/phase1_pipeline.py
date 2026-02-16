import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import MinMaxScaler
import os

os.makedirs('data/processed', exist_ok=True)


class ClimateDataPlatform:
    def __init__(self, start_year=1974, end_year=2024):
        self.start_year = start_year
        self.end_year = end_year
        self.dates = pd.date_range(
            start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='ME')

    def load_weather_data(self):
        """
        In a real scenario, use pd.read_csv('jma_data.csv').
        Here, we simulate Tokyo monthly average temperature with seasonality and a warming trend.
        """
        n = len(self.dates)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
        trend = np.linspace(0, 1.5, n)
        noise = np.random.normal(0, 0.5, n)

        temp = 15 + seasonality + trend + noise

        df_weather = pd.DataFrame({'Date': self.dates, 'Temperature': temp})
        df_weather.set_index('Date', inplace=True)
        print(f"✅ Weather data loaded: {df_weather.shape}")
        return df_weather

    def load_socioeconomic_data(self):
        """
        Socioeconomic data is usually yearly. We will generate yearly data
        and upsample it to monthly to match weather data.
        Variables: CO2 Emissions, GDP, Urbanization Rate.
        """
        years = pd.date_range(
            start=f'{self.start_year}-01-01', end=f'{self.end_year}-12-31', freq='YE')
        n = len(years)

        co2 = np.linspace(300, 420, n) + np.random.normal(0, 2, n)
        gdp = np.linspace(100, 500, n) + np.random.normal(0, 10, n)
        urbanization = np.linspace(70, 92, n) + \
            np.random.normal(0, 0.5, n)

        df_socio = pd.DataFrame({
            'Date': years,
            'CO2_Emissions': co2,
            'GDP': gdp,
            'Urbanization': urbanization
        })
        df_socio.set_index('Date', inplace=True)
        df_socio_monthly = df_socio.resample('ME').interpolate(method='cubic')

        print(f"✅ Socioeconomic data processed: {df_socio_monthly.shape}")
        return df_socio_monthly

    def merge_datasets(self, df_weather, df_socio):
        df_combined = df_weather.join(df_socio, how='inner')
        return df_combined

    def analyze_correlation(self, df):
        """
        1. Heatmap correlation
        2. Save to file
        """
        output_dir = 'reports/figures'
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))

        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix: Weather vs Socioeconomic Factors")
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'phase1_correlation_matrix.png')
        plt.savefig(save_path)
        print(f"✅ Correlation Plot saved to: {save_path}")

        plt.close()

        print("\n--- Correlation Analysis ---")
        print(corr['Temperature'].sort_values(ascending=False))

    def granger_causality_test(self, df, target='Temperature', predictor='CO2_Emissions', maxlag=12):
        """
        Statistical test to see if past values of a predictor help predict the target.
        """
        print(
            f"\n--- Granger Causality Test: Does {predictor} cause {target}? ---")

        data_subset = df[[target, predictor]]
        test_result = grangercausalitytests(
            data_subset, maxlag=maxlag, verbose=False)

        p_value = test_result[maxlag][0]['ssr_ftest'][1]
        print(f"Lag {maxlag} months P-value: {p_value:.5f}")
        if p_value < 0.05:
            print(
                f"Result: Significant evidence that {predictor} Granger-causes {target}.")
        else:
            print(f"Result: No significant evidence of causality found.")


if __name__ == "__main__":
    platform = ClimateDataPlatform()

    weather = platform.load_weather_data()
    socio = platform.load_socioeconomic_data()

    df_main = platform.merge_datasets(weather, socio)

    df_main.to_csv('data/processed/climate_socio_merged.csv')
    print("Dataset saved for Phase 2.")

    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)

    df_norm = (df_main - df_main.mean()) / df_main.std()

    plt.figure(figsize=(12, 6))
    df_norm.plot(figsize=(12, 6), title="Normalized Trends (50 Years)")

    trends_path = os.path.join(output_dir, 'phase1_trends.png')
    plt.savefig(trends_path)
    print(f"✅ Trends Plot saved to: {trends_path}")
    plt.close()

    platform.analyze_correlation(df_main)

    platform.granger_causality_test(
        df_main, target='Temperature', predictor='CO2_Emissions', maxlag=12)
    platform.granger_causality_test(
        df_main, target='Temperature', predictor='Urbanization', maxlag=12)

    print("\n✅ Phase 1 Complete. Check 'reports/figures/' for your images.")
