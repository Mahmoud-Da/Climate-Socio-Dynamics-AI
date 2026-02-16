import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

torch.manual_seed(42)
np.random.seed(42)


class ClimateDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index: index + self.seq_length]
        y = self.data[index + self.seq_length, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


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


class ClimateModelEngine:
    def __init__(self, data_path, seq_length=12, split_year='2014'):
        self.data_path = data_path
        self.seq_length = seq_length
        self.split_year = split_year
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_and_process(self):

        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        features = ['Temperature', 'CO2_Emissions', 'GDP', 'Urbanization']
        data = df[features].values

        train_mask = df.index < f'{self.split_year}-01-01'
        train_data_raw = data[train_mask]
        test_data_raw = data[~train_mask]

        self.scaler.fit(train_data_raw)
        train_scaled = self.scaler.transform(train_data_raw)
        test_scaled = self.scaler.transform(test_data_raw)

        train_dataset = ClimateDataset(train_scaled, self.seq_length)
        test_dataset = ClimateDataset(test_scaled, self.seq_length)

        return train_dataset, test_dataset, df[~train_mask].index[self.seq_length:]

    def train_model(self, train_loader, input_size, hidden_size=64, epochs=100):
        model = ClimateLSTM(input_size, hidden_size, output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("\n--- Starting Training ---")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:

                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch+1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}')

        return model

    def backtest(self, model, test_loader, test_dates):
        print("\n--- Starting Backtest (Validation) ---")
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                predictions.append(output.item())
                actuals.append(y_batch.item())

        pred_dummy = np.zeros((len(predictions), 4))
        actual_dummy = np.zeros((len(actuals), 4))

        pred_dummy[:, 0] = predictions
        actual_dummy[:, 0] = actuals

        temp_min = self.scaler.min_[0]
        temp_scale = self.scaler.scale_[0]

        real_preds = (np.array(predictions) - temp_min) / temp_scale
        real_actuals = (np.array(actuals) - temp_min) / temp_scale

        rmse = np.sqrt(np.mean((real_preds - real_actuals)**2))
        print(f"Validation RMSE: {rmse:.4f} °C")

        return real_actuals, real_preds

    def plot_results(self, dates, actuals, preds):
        os.makedirs('reports/figures', exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actuals, label='Actual Data (Hidden)',
                 color='blue', alpha=0.6)
        plt.plot(dates, preds, label='AI Model Prediction',
                 color='red', linestyle='--')
        plt.title("Phase 2 Backtest: Model Reproducibility Check")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()

        save_path = 'reports/figures/phase2_backtest.png'
        plt.savefig(save_path)
        print(f"✅ Backtest Plot saved to: {save_path}")
        plt.close()


if __name__ == "__main__":
    DATA_PATH = 'data/processed/climate_socio_merged.csv'
    SEQ_LENGTH = 12
    BATCH_SIZE = 16

    engine = ClimateModelEngine(DATA_PATH, SEQ_LENGTH)
    train_data, test_data, test_dates = engine.load_and_process()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = engine.train_model(train_loader, input_size=4, epochs=100)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/climate_lstm_v1.pth')
    print("✅ Model saved to models/climate_lstm_v1.pth")

    actuals, preds = engine.backtest(model, test_loader, test_dates)
    engine.plot_results(test_dates, actuals, preds)
