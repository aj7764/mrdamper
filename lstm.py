import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(page_title="LSTM Damper Prediction", layout="wide")
st.title("🔩 LSTM Damper – CSV Prediction Dashboard")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 50

# ==================================
# MODEL (EXACT MATCH)
# ==================================
class LSTMDamper(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

# ==================================
# LOAD ARTIFACTS
# ==================================
@st.cache_resource
def load_artifacts():
    model = LSTMDamper().to(DEVICE)
    model.load_state_dict(torch.load("lstm_damper_best.pt", map_location=DEVICE))
    model.eval()

    feat_scaler = joblib.load("feat_scaler.pkl")
    tgt_scaler = joblib.load("tgt_scaler.pkl")

    return model, feat_scaler, tgt_scaler

model, feat_scaler, tgt_scaler = load_artifacts()

# ==================================
# CSV UPLOAD
# ==================================
st.sidebar.header("📁 Upload CSV File")
uploaded_file = st.sidebar.file_uploader(
    "Upload damper CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to start prediction")
    st.stop()

# ==================================
# LOAD & VALIDATE CSV
# ==================================
df = pd.read_csv(uploaded_file)

required_cols = ["A", "D", "V", "Y", "F"]
if not all(c in df.columns for c in required_cols):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

df = df[required_cols].dropna()

# ==================================
# FEATURE ENGINEERING (MATCH TRAINING)
# ==================================
A = df["A"].values.reshape(-1, 1)
D = df["D"].values.reshape(-1, 1)
V = df["V"].values.reshape(-1, 1)
Y = df["Y"].values.reshape(-1, 1)
F_true = df["F"].values

V2 = V ** 2
VY = V * Y

X_full = np.hstack([A, D, V, Y, V2, VY])  # [N,6]

if len(X_full) < WINDOW_SIZE:
    st.error(f"CSV must have at least {WINDOW_SIZE} rows")
    st.stop()

# ==================================
# WINDOWING
# ==================================
X_windows = []
for i in range(len(X_full) - WINDOW_SIZE + 1):
    X_windows.append(X_full[i:i + WINDOW_SIZE])

X_windows = np.array(X_windows)  # [Nw, T, 6]

# Scale features
Nw, T, F = X_windows.shape
X_scaled = feat_scaler.transform(
    X_windows.reshape(-1, F)
).reshape(Nw, T, F)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

# ==================================
# PREDICTION
# ==================================
with torch.no_grad():
    y_scaled = model(X_tensor).cpu().numpy().reshape(-1, 1)

y_pred = tgt_scaler.inverse_transform(y_scaled).flatten()
y_true = F_true[WINDOW_SIZE - 1:]

# ==================================
# METRICS
# ==================================
# ===============================
# METRICS (UNSCALED, PHYSICAL)
# ===============================
mse_N2 = mean_squared_error(y_true, y_pred)        # N²
mse_kN2 = mse_N2 / 1e6                              # (kN)²
rmse_N = np.sqrt(mse_N2)
r2 = r2_score(y_true, y_pred)

st.subheader("📊 Prediction Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Samples", len(y_pred))
col2.metric("MSE (kN²)", f"{mse_kN2:.4f}")
col3.metric("RMSE (N)", f"{rmse_N:.1f}")
col4.metric("R²", f"{r2:.4f}")


# ==================================
# PLOTS
# ==================================
st.subheader("📈 Results")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. True vs Pred
axes[0, 0].scatter(y_true, y_pred, s=2, alpha=0.5)
minv, maxv = y_true.min(), y_true.max()
axes[0, 0].plot([minv, maxv], [minv, maxv], "r--")
axes[0, 0].set_title("True vs Predicted Force")
axes[0, 0].set_xlabel("True (N)")
axes[0, 0].set_ylabel("Predicted (N)")
axes[0, 0].grid(alpha=0.3)

# 2. Time series
axes[0, 1].plot(y_true, label="True", lw=1)
axes[0, 1].plot(y_pred, label="Predicted", lw=1)
axes[0, 1].set_title("Force Time Series")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. D–F hysteresis
axes[1, 0].plot(D[WINDOW_SIZE - 1:], y_true, label="Experimental", lw=1)
axes[1, 0].plot(D[WINDOW_SIZE - 1:], y_pred, label="LSTM", lw=1.5)
axes[1, 0].set_title("Displacement–Force Loop")
axes[1, 0].set_xlabel("Displacement (m)")
axes[1, 0].set_ylabel("Force (N)")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. V–F hysteresis
axes[1, 1].plot(V[WINDOW_SIZE - 1:], y_true, label="Experimental", lw=1)
axes[1, 1].plot(V[WINDOW_SIZE - 1:], y_pred, label="LSTM", lw=1.5)
axes[1, 1].set_title("Velocity–Force Loop")
axes[1, 1].set_xlabel("Velocity (m/s)")
axes[1, 1].set_ylabel("Force (N)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ==================================
# DOWNLOAD RESULTS
# ==================================
out_df = pd.DataFrame({
    "F_true": y_true,
    "F_pred": y_pred
})

st.download_button(
    "⬇️ Download Predictions CSV",
    out_df.to_csv(index=False),
    file_name="lstm_predictions.csv",
    mime="text/csv"
)
