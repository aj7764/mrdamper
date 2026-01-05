import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor")

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Damper ML Analysis Platform", layout="wide")
st.title("🔬 Damper Force Prediction Platform")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 50
MAX_PLOT_SAMPLES = 5000

# ============================================================
# MODEL DEFINITIONS
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
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.075),
            nn.Linear(64, 1),
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
# LAZY LOADERS
# ============================================================
@st.cache_resource
def load_scalers():
    return joblib.load("feat_scaler.pkl"), joblib.load("tgt_scaler.pkl")

@st.cache_resource
def load_lstm():
    m = LSTMDamper().to(DEVICE)
    m.load_state_dict(torch.load("lstm_damper_best.pt", map_location=DEVICE))
    m.eval()
    return m

@st.cache_resource
def load_pinn():
    m = PINNTransformer().to(DEVICE)
    ckpt = torch.load("pinn_transformer_best.pt", map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

@st.cache_resource
def load_transformer():
    m = ImprovedTransformer().to(DEVICE)
    m.load_state_dict(torch.load("improved_v2_best.pt", map_location=DEVICE))
    m.eval()
    return m

feat_scaler, tgt_scaler = load_scalers()

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

# Downsample for safety
if len(y_true) > MAX_PLOT_SAMPLES:
    idx = np.linspace(0, len(y_true)-1, MAX_PLOT_SAMPLES).astype(int)
    y_true = y_true[idx]
    D_plot = D_plot[idx]
    V_plot = V_plot[idx]
    X_tensor = X_tensor[idx]

def inv(y):
    return tgt_scaler.inverse_transform(y.reshape(-1,1)).flatten()

def metrics(y):
    mse = mean_squared_error(y_true, y)
    return mse/1e6, np.sqrt(mse)/1000, r2_score(y_true, y)

# ============================================================
# SINGLE MODEL ANALYSIS
# ============================================================
if mode == "Single Model Analysis":
    with torch.no_grad():
        if model_choice == "LSTM":
            model = load_lstm()
            y = inv(model(X_tensor).cpu().numpy())

        elif model_choice == "PINN":
            model = load_pinn()
            y, F_phys = model(X_tensor)
            y = inv(y.cpu().numpy())
            F_phys = F_phys.cpu().numpy()

        else:
            model = load_transformer()
            y = inv(model(X_tensor).cpu().numpy())

    mse_kN2, rmse_kN, r2 = metrics(y)

    st.subheader(f"📊 {model_choice} Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE ", f"{mse_kN2:.4f}")
    c2.metric("RMSE ", f"{rmse_kN:.4f}")
    c3.metric("R²", f"{r2:.4f}")

    fig, ax = plt.subplots(2,2, figsize=(14,10))
    ax[0,0].scatter(y_true, y, s=2)
    ax[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax[0,0].set_title("True vs Predicted")

    ax[0,1].plot(y_true, label="True")
    ax[0,1].plot(y, label="Pred")
    ax[0,1].legend()

    ax[1,0].plot(D_plot, y_true)
    ax[1,0].plot(D_plot, y)
    ax[1,0].set_title("D–F Hysteresis")

    ax[1,1].plot(V_plot, y_true)
    ax[1,1].plot(V_plot, y)
    ax[1,1].set_title("V–F Hysteresis")

    st.pyplot(fig)

    if model_choice == "PINN":
        st.subheader("⚙️ Physics Residual")
        st.line_chart(np.abs(y - F_phys))

# ============================================================
# MODEL COMPARISON
# ============================================================
else:
    with torch.no_grad():
        model = load_lstm()
        y_lstm = inv(model(X_tensor).cpu().numpy())
        del model

        model = load_pinn()
        y_pinn, F_phys = model(X_tensor)
        y_pinn = inv(y_pinn.cpu().numpy())
        F_phys = F_phys.cpu().numpy()
        del model

        model = load_transformer()
        y_tr = inv(model(X_tensor).cpu().numpy())
        del model

    st.subheader("📊 Model Comparison Metrics")
    table = pd.DataFrame({
        "LSTM": metrics(y_lstm),
        "PINN": metrics(y_pinn),
        "Transformer": metrics(y_tr),
    }, index=["MSE (kN²)", "RMSE (kN)", "R²"])
    st.dataframe(table.style.format("{:.4f}"))

    st.subheader("📈 Hysteresis Comparison")
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    ax[0].plot(D_plot, y_true, label="True")
    ax[0].plot(D_plot, y_lstm, '--', label="LSTM")
    ax[0].plot(D_plot, y_pinn, '-.', label="PINN")
    ax[0].plot(D_plot, y_tr, ':', label="Transformer")
    ax[0].legend()
    ax[0].set_title("Displacement–Force")

    ax[1].plot(V_plot, y_true, label="True")
    ax[1].plot(V_plot, y_lstm, '--', label="LSTM")
    ax[1].plot(V_plot, y_pinn, '-.', label="PINN")
    ax[1].plot(V_plot, y_tr, ':', label="Transformer")
    ax[1].legend()
    ax[1].set_title("Velocity–Force")

    st.pyplot(fig)

    st.subheader("📉 Difference Analysis")
    st.line_chart(pd.DataFrame({
        "PINN − LSTM": y_pinn - y_lstm,
        "Transformer − LSTM": y_tr - y_lstm,
    }))

    st.subheader("⚙️ PINN Physics Residual")
    st.line_chart(np.abs(y_pinn - F_phys))
