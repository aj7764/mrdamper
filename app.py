import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Damper ML Analysis Platform", layout="wide")
st.title("🔬 Damper Force Prediction Platform")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 50

# ============================================================
# MODELS
# ============================================================
class LSTMDamper(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 128, 2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class ImprovedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(6, 192)
        enc = nn.TransformerEncoderLayer(
            192, 3, 384, 0.25, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, 4)
        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        x = self.encoder(self.input_proj(x))
        return self.head(x.mean(dim=1)).squeeze(-1)


class PINNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(6, 256)
        enc = nn.TransformerEncoderLayer(
            256, 8, 512, 0.15, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, 6)
        self.head = nn.Sequential(
            nn.Linear(256, 128),      # head.0
            nn.GELU(),                # head.1
            nn.Dropout(0.15),         # head.2
            nn.Linear(128, 64),       # head.3
            nn.GELU(),                # head.4
            nn.Dropout(0.075),        # head.5  ← dropout*0.5
            nn.Linear(64, 1)          # head.6
        )
        self.boucwen = nn.ParameterDict({
            "alpha": nn.Parameter(torch.tensor(1.0)),
            "c0": nn.Parameter(torch.tensor(2.0)),
            "k0": nn.Parameter(torch.tensor(0.05)),
            "k1": nn.Parameter(torch.tensor(0.005)),
            "beta": nn.Parameter(torch.tensor(1.5)),
            "gamma": nn.Parameter(torch.tensor(0.8)),
            "n": nn.Parameter(torch.tensor(1.2)),
            "A": nn.Parameter(torch.tensor(1.2)),
        })

    def bouc_wen_force(self, xd, xv, z):
        dz = (-self.boucwen["gamma"] * torch.abs(xv) *
              torch.abs(z)**(self.boucwen["n"]-1) * z * torch.sign(xv)
              - self.boucwen["beta"] * xv * torch.abs(z)**self.boucwen["n"]
              + self.boucwen["A"] * xv)
        F = (self.boucwen["alpha"] * z +
             self.boucwen["c0"] * xv +
             self.boucwen["k0"] * xd +
             self.boucwen["k1"] * xd)
        return F, torch.clamp(z + 0.01 * dz, -10, 10)

    def forward(self, x):
        enc = self.encoder(self.input_proj(x))
        F_pred = self.head(enc.mean(dim=1)).squeeze(-1)
        xd, xv = x[:, -1, 1], x[:, -1, 2]
        F_phys, _ = self.bouc_wen_force(xd, xv, torch.zeros_like(xd))
        return F_pred, F_phys

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_all():
    feat_scaler = joblib.load("feat_scaler.pkl")
    tgt_scaler = joblib.load("tgt_scaler.pkl")

    lstm = LSTMDamper().to(DEVICE)
    lstm.load_state_dict(torch.load("lstm_damper_best.pt", map_location=DEVICE))
    lstm.eval()

    pinn = PINNTransformer().to(DEVICE)
    pinn.load_state_dict(
        torch.load("pinn_transformer_best.pt", map_location=DEVICE)["model_state_dict"]
    )
    pinn.eval()

    tr = ImprovedTransformer().to(DEVICE)
    tr.load_state_dict(torch.load("improved_v2_best.pt", map_location=DEVICE))
    tr.eval()

    return lstm, pinn, tr, feat_scaler, tgt_scaler

lstm, pinn, transformer, feat_scaler, tgt_scaler = load_all()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["Single Model Analysis", "Model Comparison"])
model_choice = None
if mode == "Single Model Analysis":
    model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "PINN", "Transformer"])

csv = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
if csv is None:
    st.stop()

# ============================================================
# PREPROCESS
# ============================================================
df = pd.read_csv(csv)[["A", "D", "V", "Y", "F"]].dropna()
A, D, V, Y, F = [df[c].values.reshape(-1, 1) for c in ["A", "D", "V", "Y", "F"]]
X = np.hstack([A, D, V, Y, V**2, V*Y])

Xw = np.array([X[i:i+WINDOW_SIZE] for i in range(len(X)-WINDOW_SIZE+1)])
Xw = feat_scaler.transform(Xw.reshape(-1, 6)).reshape(Xw.shape)
X_tensor = torch.tensor(Xw, dtype=torch.float32).to(DEVICE)

y_true = F[WINDOW_SIZE-1:].flatten()
D_plot = D[WINDOW_SIZE-1:].flatten()
V_plot = V[WINDOW_SIZE-1:].flatten()

# ============================================================
# PREDICTIONS
# ============================================================
with torch.no_grad():
    y_lstm = lstm(X_tensor).cpu().numpy()
    y_tr = transformer(X_tensor).cpu().numpy()
    y_pinn, F_phys = pinn(X_tensor)
    y_pinn, F_phys = y_pinn.cpu().numpy(), F_phys.cpu().numpy()

def inv(y): return tgt_scaler.inverse_transform(y.reshape(-1,1)).flatten()

y_lstm, y_tr, y_pinn = inv(y_lstm), inv(y_tr), inv(y_pinn)

# ============================================================
# METRICS
# ============================================================
def metrics(y):
    mse = mean_squared_error(y_true, y)
    return mse/1e6, np.sqrt(mse)/1000, r2_score(y_true, y)

# ============================================================
# SINGLE MODEL VIEW
# ============================================================
if mode == "Single Model Analysis":
    y_map = {"LSTM": y_lstm, "PINN": y_pinn, "Transformer": y_tr}
    y = y_map[model_choice]

    mse_kN2, rmse_kN, r2 = metrics(y)

    st.subheader(f"📊 {model_choice} Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE (kN²)", f"{mse_kN2:.4f}")
    c2.metric("RMSE (kN)", f"{rmse_kN:.4f}")
    c3.metric("R²", f"{r2:.4f}")

    fig, ax = plt.subplots(2,2, figsize=(14,10))
    ax[0,0].scatter(y_true, y, s=2); ax[0,0].plot([y_true.min(), y_true.max()],[y_true.min(), y_true.max()],'r--')
    ax[0,0].set_title("True vs Predicted")

    ax[0,1].plot(y_true, label="True"); ax[0,1].plot(y, label="Pred"); ax[0,1].legend()

    ax[1,0].plot(D_plot, y_true); ax[1,0].plot(D_plot, y); ax[1,0].set_title("D–F Hysteresis")
    ax[1,1].plot(V_plot, y_true); ax[1,1].plot(V_plot, y); ax[1,1].set_title("V–F Hysteresis")
    st.pyplot(fig)

    if model_choice == "PINN":
        st.subheader("⚙️ Physics Residual")
        st.line_chart(np.abs(y_pinn - F_phys))

# ============================================================
# COMPARISON VIEW
# ============================================================
else:
    st.subheader("📊 Model Comparison Metrics")
    table = pd.DataFrame({
        "LSTM": metrics(y_lstm),
        "PINN": metrics(y_pinn),
        "Transformer": metrics(y_tr),
    }, index=["MSE (kN²)", "RMSE (kN)", "R²"])
    st.dataframe(table.style.format("{:.4f}"))

    st.subheader("📉 Difference Analysis")
    st.line_chart(pd.DataFrame({
        "PINN − LSTM": y_pinn - y_lstm,
        "Transformer − LSTM": y_tr - y_lstm,
        "Transformer − PINN": y_tr - y_pinn,
    }))

    st.subheader("⚙️ PINN Physics Residual")
    st.line_chart(np.abs(y_pinn - F_phys))
    # ============================================================
    # OVERLAY PLOTS (COMPARISON)
    # ============================================================
    st.subheader("📈 Model Comparison – Prediction & Hysteresis")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ------------------------------------------------------------
    # 1. Force Time Series
    # ------------------------------------------------------------
    axes[0, 0].plot(y_true, label="Experimental", lw=1)
    axes[0, 0].plot(y_lstm, label="LSTM", lw=1)
    axes[0, 0].plot(y_pinn, label="PINN", lw=1)
    axes[0, 0].plot(y_tr, label="Transformer", lw=1)
    axes[0, 0].set_title("Force Time Series")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Force (N)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # ------------------------------------------------------------
    # 2. True vs Predicted (scatter)
    # ------------------------------------------------------------
    axes[0, 1].scatter(y_true, y_lstm, s=2, alpha=0.3, label="LSTM")
    axes[0, 1].scatter(y_true, y_pinn, s=2, alpha=0.3, label="PINN")
    axes[0, 1].scatter(y_true, y_tr, s=2, alpha=0.3, label="Transformer")
    mn, mx = y_true.min(), y_true.max()
    axes[0, 1].plot([mn, mx], [mn, mx], "r--")
    axes[0, 1].set_title("True vs Predicted Force")
    axes[0, 1].set_xlabel("True Force (N)")
    axes[0, 1].set_ylabel("Predicted Force (N)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # ------------------------------------------------------------
    # 3. Displacement–Force Hysteresis
    # ------------------------------------------------------------
    axes[1, 0].plot(D_plot, y_true, lw=1, label="Experimental")
    axes[1, 0].plot(D_plot, y_lstm, lw=1.3, linestyle="--", label="LSTM")
    axes[1, 0].plot(D_plot, y_pinn, lw=1.3, linestyle="-.", label="PINN")
    axes[1, 0].plot(D_plot, y_tr, lw=1.3, linestyle=":", label="Transformer")
    axes[1, 0].set_title("Displacement–Force Hysteresis")
    axes[1, 0].set_xlabel("Displacement (m)")
    axes[1, 0].set_ylabel("Force (N)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # ------------------------------------------------------------
    # 4. Velocity–Force Hysteresis
    # ------------------------------------------------------------
    axes[1, 1].plot(V_plot, y_true, lw=1, label="Experimental")
    axes[1, 1].plot(V_plot, y_lstm, lw=1.3, linestyle="--", label="LSTM")
    axes[1, 1].plot(V_plot, y_pinn, lw=1.3, linestyle="-.", label="PINN")
    axes[1, 1].plot(V_plot, y_tr, lw=1.3, linestyle=":", label="Transformer")
    axes[1, 1].set_title("Velocity–Force Hysteresis")
    axes[1, 1].set_xlabel("Velocity (m/s)")
    axes[1, 1].set_ylabel("Force (N)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
