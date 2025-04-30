import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Hyperparameters ---
window_size_learning = 200 #Sliding window learning
window_size_inference = 200 #Sliding window inference
params = {'Em': 1, 'Tau_e': 50, 'Tau_s': 25, 'EmNL': 0.2, 'Alpha': 3} #Parameters for the Zener model
Nsamp = 100
Nepoch = 100
batch_size=32
hidden_size = 64
num_layers = 2

# --- Data Generator ---
class data_generator:
    def __init__(self, Rsig, Nsamp):
        self.Np = len(Rsig)
        self.Nsamp = Nsamp
        self.Sig0 = 1
        self.t0 = 0.0
        self.Rsig = Rsig

    def fSig(self, x, x1, Rs):
        return Rs * (x - x1)

    def fEpsNL(self, Sig, EmNL, Alpha):
        return Sig / (EmNL * (1 + Alpha * Sig))

    def fEps(self, x, x1, x2, Rs, Em, Tau_s, Tau_e):
        return (Rs / Em) * (
            (x2 - x1) + (Tau_s - Tau_e) * (np.exp(x2 / Tau_e) - np.exp(x1 / Tau_e)) * np.exp(-x / Tau_e)
        )

    def fEps0(self, x, Rs, Em, Tau_s, Tau_e):
        return self.fEps(x, self.t0, x, Rs, Em, Tau_s, Tau_e)

    def fEpsVE(self, x, i, Rsig, R, Tp, Em, Tau_s, Tau_e):
        epsVE = self.fEps0(x - (i - 1) * Tp, Rsig[i - 1] * R, Em, Tau_s, Tau_e)
        for k in range(1, i):
            epsVE += self.fEps(x, (k - 1) * Tp, k * Tp, Rsig[k - 1] * R, Em, Tau_s, Tau_e)
        return epsVE

    def compute_all(self, Em, Tau_e, Tau_s, EmNL, Alpha, dt=0.1):
        t = 0.0
        FvalSig = 0.0
        Tp = dt * self.Nsamp
        R = self.Sig0 / Tp
        data = []

        for i in range(1, self.Np + 1):
            for j in range(self.Nsamp):
                t += dt
                valSig = FvalSig + self.fSig(t, (i - 1) * Tp, self.Rsig[i - 1] * R)
                valEpsNL = self.fEpsNL(valSig, EmNL, Alpha)
                valEpsVE = self.fEpsVE(t, i, self.Rsig, R, Tp, Em, Tau_s, Tau_e)
                data.append([
                    t, valSig, valEpsNL, valEpsVE, valEpsNL + valEpsVE
                ])
                
            FvalSig = valSig

        return pd.DataFrame(data, columns=[
            'time', 'Stress', 'StrainNL', 'StrainVE', 'TotStrain'])

# --- Data Preparation ---
signals = [[+1, 0, -1, +1, 0, -1, +1, 0, -1],
           [+1, +1, 0, 0, -1, -1, +1, 0, -1],
           [+1, +1, +1, 0, 0, 0, -1, -1, -1]]

data = []
for sig in signals:
    gen = data_generator(sig, Nsamp)
    df = gen.compute_all(**params)
    data.append(df)

# --- Sliding Window ---
X_list, y_list = [], []
for df in data:
    for i in range(len(df) - window_size_learning):
        stress_window = df['Stress'].iloc[i:i+window_size_learning].to_numpy().reshape(window_size_learning, 1)
        strain_target = df[['StrainNL', 'StrainVE']].iloc[i+window_size_learning-1].to_numpy()  # last step
        X_list.append(stress_window)
        y_list.append(strain_target)

X = torch.tensor(np.array(X_list), dtype=torch.float32)  # (N, window, 1)
y = torch.tensor(np.array(y_list), dtype=torch.float32)  # (N, 2)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_output = out[:, -1, :]  # Take only last time step
        return self.fc(last_output)

model = LSTMModel(hidden_size=hidden_size,num_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- Training ---
for epoch in range(Nepoch):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

# --- Evaluation ---
new_Rsig = [+1,0,-1, +1,0,-1, +1,0,-1,  +1, +1, 0, 0,-1,-1,  +1, +1, 0, 0,-1,-1 ]
gen = data_generator(new_Rsig, Nsamp)
test_df = gen.compute_all(**params)

# Prepare test input (sliding windows)
stress_seq = torch.tensor(test_df['Stress'].to_numpy(), dtype=torch.float32)
true_nl = test_df['StrainNL'].to_numpy()
true_ve = test_df['StrainVE'].to_numpy()
true_tot = test_df['TotStrain'].to_numpy()

test_X = []
for i in range(len(stress_seq) - window_size_inference):
    window_stress = stress_seq[i:i+window_size_inference].unsqueeze(1)
    test_X.append(window_stress)

test_X = torch.stack(test_X)

# --- Prediction ---
model.eval()
with torch.no_grad():
    preds = model(test_X)
    pred_nl = preds[:, 0].numpy()
    pred_ve = preds[:, 1].numpy()
    pred_total = pred_nl + pred_ve

# --- Plot ---
t_test = test_df['time'].to_numpy()[window_size_inference:]

plt.figure(figsize=(10, 5))
plt.plot(t_test, true_tot[window_size_inference:], label="True Total Strain", linewidth=2)
plt.plot(t_test, pred_total, label="Predicted Total Strain", linestyle='--')
plt.plot(t_test, true_nl[window_size_inference:], label="True StrainNL", alpha=0.5)
plt.plot(t_test, pred_nl, label="Predicted StrainNL", linestyle='--', alpha=0.5)
plt.plot(t_test, true_ve[window_size_inference:], label="True StrainVE", alpha=0.5)
plt.plot(t_test, pred_ve, label="Predicted StrainVE", linestyle='--', alpha=0.5)
plt.title("Strain Prediction from Stress (No Parameters)")
plt.xlabel("Time (s)")
plt.ylabel("Strain")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
