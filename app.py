import time
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", message="enable_nested_tensor")

st.set_page_config(
    page_title="Damper ML Analysis Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 50
MAX_PLOT_SAMPLES = 5000
MODEL_COLORS = {
    "Actual": "#102542",
    "LSTM": "#f97316",
    "PINN": "#0f766e",
    "Transformer": "#7c3aed",
}
PLOT_TEMPLATE = "plotly_white"


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #f6f1e8;
            --bg-card: rgba(255, 255, 255, 0.80);
            --border-soft: rgba(16, 37, 66, 0.10);
            --ink-strong: #102542;
            --ink-soft: #516176;
            --accent: #d97706;
            --accent-2: #0f766e;
            --accent-3: #7c3aed;
            --shadow-soft: 0 22px 60px rgba(16, 37, 66, 0.10);
        }

        html, body, [class*="css"]  {
            font-family: "Manrope", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 119, 6, 0.18), transparent 26%),
                radial-gradient(circle at 85% 15%, rgba(15, 118, 110, 0.16), transparent 20%),
                linear-gradient(180deg, #fbf8f2 0%, var(--bg-main) 100%);
            color: var(--ink-strong);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }

        header[data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 251, 245, 0.98) 0%, rgba(248, 243, 235, 0.96) 100%);
            border-right: 1px solid var(--border-soft);
        }

        [data-testid="stSidebar"] > div:first-child {
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--ink-strong);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
            color: var(--ink-soft) !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] label {
            font-weight: 700;
            color: var(--ink-strong) !important;
        }

        [data-testid="stSidebar"] .stAlert {
            background: rgba(16, 37, 66, 0.06);
            border: 1px solid rgba(16, 37, 66, 0.10);
            color: var(--ink-strong);
        }

        .sidebar-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.84) 0%, rgba(255,255,255,0.70) 100%);
            border: 1px solid rgba(16, 37, 66, 0.10);
            border-radius: 24px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 16px 36px rgba(16, 37, 66, 0.08);
            margin-bottom: 1rem;
        }

        .sidebar-kicker {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: rgba(16, 37, 66, 0.08);
            color: var(--ink-strong);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }

        .sidebar-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--ink-strong);
            margin-bottom: 0.3rem;
        }

        .sidebar-copy {
            color: var(--ink-soft);
            font-size: 0.92rem;
            line-height: 1.5;
            margin: 0;
        }

        .sidebar-section-label {
            color: var(--ink-strong);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin: 0.85rem 0 0.4rem 0;
        }

        .quick-guide {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(16, 37, 66, 0.08);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 14px 34px rgba(16, 37, 66, 0.07);
        }

        .quick-guide-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: var(--ink-strong);
            margin-bottom: 0.35rem;
        }

        .quick-guide-copy {
            color: var(--ink-soft);
            margin: 0;
        }

        .stat-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(16, 37, 66, 0.08);
            border-radius: 22px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 12px 30px rgba(16, 37, 66, 0.07);
            min-height: 126px;
        }

        .stat-card.compact {
            min-height: 108px;
        }

        .stat-label {
            color: var(--ink-soft);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .stat-value {
            color: var(--ink-strong);
            font-size: 1.45rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .stat-value.large {
            font-size: 2.05rem;
        }

        .stat-copy {
            color: var(--ink-soft);
            font-size: 0.95rem;
            line-height: 1.45;
            margin: 0;
        }

        .highlight-banner {
            background: linear-gradient(135deg, rgba(16, 37, 66, 0.96), rgba(15, 118, 110, 0.92));
            color: white;
            border-radius: 22px;
            padding: 1rem 1.15rem;
            box-shadow: 0 14px 32px rgba(16, 37, 66, 0.12);
            margin: 0.85rem 0 1rem 0;
        }

        .highlight-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 800;
            opacity: 0.75;
            margin-bottom: 0.35rem;
        }

        .highlight-title {
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 0.18rem;
        }

        .highlight-copy {
            margin: 0;
            color: rgba(255,255,255,0.82);
        }

        .subtle-note {
            color: var(--ink-soft);
            font-size: 0.92rem;
            margin: 0.1rem 0 0.9rem 0;
        }

        .stDataFrame, .stTable {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(16, 37, 66, 0.08);
            border-radius: 18px;
            overflow: hidden;
        }

        .hero-shell {
            background:
                linear-gradient(135deg, rgba(16, 37, 66, 0.96) 0%, rgba(18, 74, 90, 0.92) 58%, rgba(217, 119, 6, 0.90) 100%);
            border-radius: 28px;
            padding: 2rem 2rem 1.7rem 2rem;
            color: white;
            box-shadow: var(--shadow-soft);
            overflow: hidden;
            position: relative;
            margin-bottom: 1.25rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -80px -120px auto;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.10);
            filter: blur(2px);
        }

        .eyebrow {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.15);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2rem, 3vw, 3.4rem);
            line-height: 1.02;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            color: rgba(255, 255, 255, 0.82);
            max-width: 760px;
            font-size: 1rem;
            margin-top: 0.85rem;
            margin-bottom: 0;
        }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 22px;
            box-shadow: var(--shadow-soft);
            padding: 1.15rem 1.2rem;
            backdrop-filter: blur(10px);
        }

        .info-card {
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid var(--border-soft);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 126px;
            box-shadow: 0 10px 28px rgba(16, 37, 66, 0.06);
        }

        .info-label {
            color: var(--ink-soft);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .info-value {
            color: var(--ink-strong);
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }

        .info-caption {
            color: var(--ink-soft);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .section-title {
            color: var(--ink-strong);
            font-size: 1.15rem;
            font-weight: 800;
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--ink-soft);
            margin-bottom: 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 18px rgba(16, 37, 66, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--ink-soft);
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink-strong);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.60);
            border: 1px solid rgba(16, 37, 66, 0.08);
            color: var(--ink-strong);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(217, 119, 6, 0.16), rgba(15, 118, 110, 0.14));
        }

        .stDownloadButton button,
        .stButton button {
            border-radius: 999px;
            border: 1px solid transparent;
            background: linear-gradient(135deg, #102542 0%, #124a5a 100%);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1.1rem;
        }

        .stFileUploader {
            background: rgba(255, 255, 255, 0.50);
            border-radius: 18px;
            padding: 0.35rem;
            border: 1px dashed rgba(16, 37, 66, 0.20);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        self.boucwen = nn.ParameterDict(
            {
                "alpha": nn.Parameter(torch.tensor(1.0)),
                "c0": nn.Parameter(torch.tensor(2.0)),
                "k0": nn.Parameter(torch.tensor(0.05)),
                "k1": nn.Parameter(torch.tensor(0.005)),
                "beta": nn.Parameter(torch.tensor(1.5)),
                "gamma": nn.Parameter(torch.tensor(0.8)),
                "n": nn.Parameter(torch.tensor(1.2)),
                "A": nn.Parameter(torch.tensor(1.2)),
            }
        )

    def bouc_wen_force(self, xd, xv, z):
        dz = (
            -self.boucwen["gamma"]
            * torch.abs(xv)
            * torch.abs(z) ** (self.boucwen["n"] - 1)
            * z
            * torch.sign(xv)
            - self.boucwen["beta"] * xv * torch.abs(z) ** self.boucwen["n"]
            + self.boucwen["A"] * xv
        )
        force = (
            self.boucwen["alpha"] * z
            + self.boucwen["c0"] * xv
            + self.boucwen["k0"] * xd
            + self.boucwen["k1"] * xd
        )
        return force, torch.clamp(z + 0.01 * dz, -10, 10)

    def forward(self, x):
        enc = self.encoder(self.input_proj(x))
        force_pred = self.head(enc.mean(dim=1)).squeeze(-1)
        xd, xv = x[:, -1, 1], x[:, -1, 2]
        force_phys, _ = self.bouc_wen_force(xd, xv, torch.zeros_like(xd))
        return force_pred, force_phys


@st.cache_resource
def load_scalers():
    return joblib.load("feat_scaler.pkl"), joblib.load("tgt_scaler.pkl")


@st.cache_resource
def load_lstm():
    model = LSTMDamper().to(DEVICE)
    model.load_state_dict(torch.load("lstm_damper_best.pt", map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_pinn():
    model = PINNTransformer().to(DEVICE)
    checkpoint = torch.load("pinn_transformer_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_transformer():
    model = ImprovedTransformer().to(DEVICE)
    model.load_state_dict(torch.load("improved_v2_best.pt", map_location=DEVICE))
    model.eval()
    return model


def render_hero():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="eyebrow">Structural Intelligence Suite</div>
            <h1 class="hero-title">Damper Force Prediction Platform</h1>
            <p class="hero-subtitle">
                A cleaner workspace for exploring force-response behavior, comparing learned models,
                and exporting polished analysis outputs from uploaded damper test data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(label, value, caption):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label, value, caption, large=False, compact=False):
    large_class = " large" if large else ""
    compact_class = " compact" if compact else ""
    st.markdown(
        f"""
        <div class="stat-card{compact_class}">
            <div class="stat-label">{label}</div>
            <div class="stat-value{large_class}">{value}</div>
            <p class="stat-copy">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title, copy):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_highlight_banner(kicker, title, copy):
    st.markdown(
        f"""
        <div class="highlight-banner">
            <div class="highlight-kicker">{kicker}</div>
            <div class="highlight-title">{title}</div>
            <p class="highlight-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_intro():
    st.sidebar.markdown(
        """
        <div class="sidebar-shell">
            <div class="sidebar-kicker">Control Deck</div>
            <div class="sidebar-title">Configure the analysis</div>
            <p class="sidebar-copy">
                Choose the evaluation mode, pick a model focus, and upload a damper CSV to unlock the visual workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section_label(text):
    st.sidebar.markdown(
        f'<div class="sidebar-section-label">{text}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state_banner():
    st.markdown(
        """
        <div class="quick-guide">
            <div class="quick-guide-title">Start with a damper CSV</div>
            <p class="quick-guide-copy">
                Upload a file from the control deck to reveal model metrics, force traces, hysteresis loops, diagnostics, and export-ready results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def configure_sidebar():
    render_sidebar_intro()
    sidebar_section_label("Workspace Mode")
    mode = st.sidebar.radio(
        "Workspace mode",
        ["Single Model Analysis", "Model Comparison"],
    )

    model_choice = None
    if mode == "Single Model Analysis":
        sidebar_section_label("Model Focus")
        model_choice = st.sidebar.selectbox(
            "Focus model",
            ["LSTM", "PINN", "Transformer"],
        )

    sidebar_section_label("Upload Test Data")
    csv_file = st.sidebar.file_uploader("Damper CSV file", type=["csv"])

    model_descriptions = {
        "LSTM": "Sequence model tuned for temporal hysteresis behavior.",
        "PINN": "Hybrid predictor with a Bouc-Wen-inspired physics constraint.",
        "Transformer": "Attention-based model for broader sequence context.",
    }
    active_model = model_choice or "Comparison Suite"
    detail_text = (
        model_descriptions.get(model_choice, "Benchmarks all available models side by side.")
        if model_choice
        else "Benchmarks latency and accuracy across all loaded predictors."
    )

    sidebar_section_label("Session Context")
    st.sidebar.info(f"**Active view:** {active_model}\n\n{detail_text}")
    st.sidebar.caption(
        f"Running on `{DEVICE.upper()}` with a rolling window of `{WINDOW_SIZE}` samples."
    )
    return mode, model_choice, csv_file


def load_and_prepare_data(csv_file, feat_scaler):
    df = pd.read_csv(csv_file)
    required_cols = ["A", "D", "V", "Y", "F"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"CSV is missing required columns: {', '.join(missing)}")
        st.stop()

    df = df[required_cols].dropna()
    if len(df) < WINDOW_SIZE:
        st.error(f"CSV must contain at least {WINDOW_SIZE} rows after removing blanks.")
        st.stop()

    a, d, v, y, f = [df[col].values.reshape(-1, 1) for col in required_cols]
    features = np.hstack([a, d, v, y, v**2, v * y])
    windows = np.array(
        [features[i : i + WINDOW_SIZE] for i in range(len(features) - WINDOW_SIZE + 1)]
    )
    windows = feat_scaler.transform(windows.reshape(-1, 6)).reshape(windows.shape)
    x_tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)

    y_true = f[WINDOW_SIZE - 1 :].flatten()
    d_plot = d[WINDOW_SIZE - 1 :].flatten()
    v_plot = v[WINDOW_SIZE - 1 :].flatten()

    if len(y_true) > MAX_PLOT_SAMPLES:
        idx = np.linspace(0, len(y_true) - 1, MAX_PLOT_SAMPLES).astype(int)
        y_true = y_true[idx]
        d_plot = d_plot[idx]
        v_plot = v_plot[idx]
        x_tensor = x_tensor[idx]

    summary = {
        "raw_rows": len(df),
        "samples": len(y_true),
        "max_force": float(df["F"].max()),
        "max_velocity": float(df["V"].max()),
        "max_displacement": float(df["D"].max()),
    }
    return df, x_tensor, y_true, d_plot, v_plot, summary


def inverse_transform_predictions(values, tgt_scaler):
    return tgt_scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    safe_denominator = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / safe_denominator)) * 100
    return {
        "MSE (kN²)": mse / 1e6,
        "RMSE (kN)": rmse / 1000,
        "MAE (kN)": mae / 1000,
        "MAPE (%)": mape,
        "R²": r2_score(y_true, y_pred),
    }


def download_csv_button(dataframe, filename, label):
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def style_plotly(fig, title=None, height=460):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=18, r=190, t=56 if title else 22, b=18),
        paper_bgcolor="rgba(255,255,255,0.72)",
        plot_bgcolor="rgba(255,255,255,0.90)",
        font=dict(color="#102542", family="Manrope, sans-serif", size=14),
        title=dict(
            text=title,
            font=dict(color="#102542", size=24, family="Manrope, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            valign="top",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(16, 37, 66, 0.12)",
            borderwidth=1,
            font=dict(color="#102542", size=14),
            title=dict(
                text="<b>Legend</b>",
                font=dict(color="#102542", size=14),
            ),
        ),
        hoverlabel=dict(
            bgcolor="rgba(16, 37, 66, 0.96)",
            font=dict(color="#ffffff", size=13),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(16, 37, 66, 0.08)",
        title_font=dict(color="#102542", size=16),
        tickfont=dict(color="#516176", size=12),
        zerolinecolor="rgba(16, 37, 66, 0.18)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(16, 37, 66, 0.08)",
        title_font=dict(color="#102542", size=16),
        tickfont=dict(color="#516176", size=12),
        zerolinecolor="rgba(16, 37, 66, 0.18)",
    )
    fig.update_annotations(font=dict(color="#102542", size=15, family="Manrope, sans-serif"))
    return fig


def sample_series(*arrays, max_points=1600):
    if not arrays:
        return tuple()
    length = len(arrays[0])
    step = max(1, length // max_points)
    return tuple(arr[::step] for arr in arrays)


def build_prediction_overview_figure(y_true, y_pred, d_plot, v_plot, model_name):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "True vs Predicted",
            "Force Trace",
            "Displacement-Force Loop",
            "Velocity-Force Loop",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name=f"<b>{model_name} Points</b>",
            marker=dict(size=5, color=MODEL_COLORS[model_name], opacity=0.55),
        ),
        row=1,
        col=1,
    )
    diag_min = float(min(y_true.min(), y_pred.min()))
    diag_max = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[diag_min, diag_max],
            y=[diag_min, diag_max],
            mode="lines",
            name="<b>Ideal Fit</b>",
            line=dict(color="#475569", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(y_true)),
            y=y_true,
            mode="lines",
            name="<b>Actual</b>",
            line=dict(color=MODEL_COLORS["Actual"], width=2.2),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(y_pred)),
            y=y_pred,
            mode="lines",
            name=f"<b>{model_name}</b>",
            line=dict(color=MODEL_COLORS[model_name], width=2),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=d_plot,
            y=y_true,
            mode="lines",
            name="<b>Actual</b>",
            line=dict(color=MODEL_COLORS["Actual"], width=2.2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d_plot,
            y=y_pred,
            mode="lines",
            name=f"<b>{model_name}</b>",
            line=dict(color=MODEL_COLORS[model_name], width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=v_plot,
            y=y_true,
            mode="lines",
            name="<b>Actual</b>",
            line=dict(color=MODEL_COLORS["Actual"], width=2.2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=v_plot,
            y=y_pred,
            mode="lines",
            name=f"<b>{model_name}</b>",
            line=dict(color=MODEL_COLORS[model_name], width=2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="True Force (N)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Force (N)", row=1, col=1)
    fig.update_xaxes(title_text="Sample", row=1, col=2)
    fig.update_yaxes(title_text="Force (N)", row=1, col=2)
    fig.update_xaxes(title_text="Displacement (m)", row=2, col=1)
    fig.update_yaxes(title_text="Force (N)", row=2, col=1)
    fig.update_xaxes(title_text="Velocity (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Force (N)", row=2, col=2)
    return style_plotly(fig, f"{model_name} Analysis Suite", height=760)


def build_residual_chart(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    fig = px.area(
        x=np.arange(len(residuals)),
        y=np.abs(residuals),
        labels={"x": "Sample", "y": "Absolute Error (N)"},
    )
    fig.update_traces(
        name=f"<b>{model_name} Absolute Error</b>",
        showlegend=True,
        line_color=MODEL_COLORS[model_name],
        fillcolor="rgba(217, 119, 6, 0.18)" if model_name == "LSTM" else None,
    )
    return style_plotly(fig, f"{model_name} Residual Envelope", height=350)


def build_physics_chart(y_pred, force_phys):
    fig = px.line(
        x=np.arange(len(y_pred)),
        y=np.abs(y_pred - force_phys),
        labels={"x": "Sample", "y": "|Predicted - Physics|"},
    )
    fig.update_traces(
        name="<b>PINN Physics Residual</b>",
        showlegend=True,
        line=dict(color=MODEL_COLORS["PINN"], width=2.4),
    )
    return style_plotly(fig, "PINN Physics Residual", height=350)


def build_comparison_timeseries(y_true, predictions):
    plot_y_true, plot_lstm, plot_pinn, plot_transformer = sample_series(
        y_true,
        predictions["LSTM"],
        predictions["PINN"],
        predictions["Transformer"],
        max_points=2200,
    )
    plot_predictions = {
        "LSTM": plot_lstm,
        "PINN": plot_pinn,
        "Transformer": plot_transformer,
    }
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(plot_y_true)),
            y=plot_y_true,
            mode="lines",
            name="<b>Actual</b>",
            line=dict(color=MODEL_COLORS["Actual"], width=2.5),
        )
    )
    for name, values in plot_predictions.items():
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(values)),
                y=values,
                mode="lines",
                name=f"<b>{name}</b>",
                line=dict(color=MODEL_COLORS[name], width=1.8),
            )
        )
    fig.update_xaxes(title_text="Sample")
    fig.update_yaxes(title_text="Force (N)")
    return style_plotly(fig, "Force Trace Comparison", height=420)


def build_error_distribution(y_true, predictions):
    fig = go.Figure()
    for name, values in predictions.items():
        fig.add_trace(
            go.Histogram(
                x=y_true - values,
                name=f"<b>{name}</b>",
                opacity=0.55,
                nbinsx=50,
                marker_color=MODEL_COLORS[name],
            )
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Prediction Error (N)")
    fig.update_yaxes(title_text="Count")
    return style_plotly(fig, "Error Distribution", height=400)


def build_hysteresis_comparison(d_plot, v_plot, y_true, predictions):
    plot_d, plot_v, plot_y_true, plot_lstm, plot_pinn, plot_transformer = sample_series(
        d_plot,
        v_plot,
        y_true,
        predictions["LSTM"],
        predictions["PINN"],
        predictions["Transformer"],
        max_points=1400,
    )
    plot_predictions = {
        "LSTM": plot_lstm,
        "PINN": plot_pinn,
        "Transformer": plot_transformer,
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Displacement-Force", "Velocity-Force"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_d,
            y=plot_y_true,
            mode="lines",
            name="<b>Actual</b>",
            line=dict(color=MODEL_COLORS["Actual"], width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_v,
            y=plot_y_true,
            mode="lines",
            name="<b>Actual</b>",
            line=dict(color=MODEL_COLORS["Actual"], width=2.4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for name, values in plot_predictions.items():
        fig.add_trace(
            go.Scatter(
                x=plot_d,
                y=values,
                mode="lines",
                name=f"<b>{name}</b>",
                line=dict(color=MODEL_COLORS[name], width=1.7, dash="dot"),
                opacity=0.82,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_v,
                y=values,
                mode="lines",
                name=f"<b>{name}</b>",
                line=dict(color=MODEL_COLORS[name], width=1.7, dash="dot"),
                opacity=0.82,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Displacement (m)", row=1, col=1)
    fig.update_yaxes(title_text="Force (N)", row=1, col=1)
    fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Force (N)", row=1, col=2)
    return style_plotly(fig, "Hysteresis Comparison", height=470)


def build_interactive_hysteresis(d_plot, y_true, predictions):
    plot_d, plot_y_true, plot_lstm, plot_pinn, plot_transformer = sample_series(
        d_plot,
        y_true,
        predictions["LSTM"],
        predictions["PINN"],
        predictions["Transformer"],
        max_points=1400,
    )
    plot_df = pd.DataFrame(
        {
            "Displacement": plot_d,
            "Actual": plot_y_true,
            "LSTM": plot_lstm,
            "PINN": plot_pinn,
            "Transformer": plot_transformer,
        }
    )
    fig = px.line(
        plot_df,
        x="Displacement",
        y=list(plot_df.columns[1:]),
        labels={"value": "Force (N)", "variable": "Series"},
        color_discrete_map={
            "Actual": MODEL_COLORS["Actual"],
            "LSTM": MODEL_COLORS["LSTM"],
            "PINN": MODEL_COLORS["PINN"],
            "Transformer": MODEL_COLORS["Transformer"],
        },
    )
    fig.for_each_trace(lambda trace: trace.update(name=f"<b>{trace.name}</b>"))
    return style_plotly(fig, "Interactive Displacement Sweep", height=460)


def build_difference_chart(predictions):
    plot_pinn, plot_lstm, plot_transformer = sample_series(
        predictions["PINN"],
        predictions["LSTM"],
        predictions["Transformer"],
        max_points=2200,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(plot_pinn)),
            y=plot_pinn - plot_lstm,
            mode="lines",
            name="<b>PINN - LSTM</b>",
            line=dict(color=MODEL_COLORS["PINN"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(plot_transformer)),
            y=plot_transformer - plot_lstm,
            mode="lines",
            name="<b>Transformer - LSTM</b>",
            line=dict(color=MODEL_COLORS["Transformer"], width=2),
        )
    )
    fig.update_xaxes(title_text="Sample")
    fig.update_yaxes(title_text="Force Difference (N)")
    return style_plotly(fig, "Difference Analysis", height=360)


def render_dataset_summary(summary, mode, model_choice):
    card1, card2, card3, card4 = st.columns(4)
    with card1:
        render_stat_card(
            "Workspace Mode",
            "Single" if mode == "Single Model Analysis" else "Compare",
            "Focused inspection for one model or a full benchmark suite.",
            large=True,
        )
    with card2:
        render_stat_card(
            "Focus",
            model_choice or "All Models",
            "Current selection driving the analysis canvas.",
            large=True,
        )
    with card3:
        render_stat_card(
            "Samples",
            f"{summary['samples']:,}",
            "Post-windowing samples used in evaluation and plotting.",
            large=True,
        )
    with card4:
        render_stat_card(
            "Runtime",
            DEVICE.upper(),
            "Inference device detected for this session.",
            large=True,
        )

    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        render_stat_card("Peak Force (N)", f"{summary['max_force']:.2f}", "Highest recorded force in the uploaded run.", compact=True)
    with stat2:
        render_stat_card("Peak Velocity (m/s)", f"{summary['max_velocity']:.2f}", "Velocity ceiling observed in the dataset.", compact=True)
    with stat3:
        render_stat_card("Peak Displacement (m)", f"{summary['max_displacement']:.2f}", "Maximum displacement covered by the uploaded sweep.", compact=True)


def render_empty_state():
    render_empty_state_banner()
    left, right, extra = st.columns(3)
    with left:
        render_info_card(
            "Expected Columns",
            "A, D, V, Y, F",
            "Acceleration, displacement, velocity, control variable, and force.",
        )
    with right:
        render_info_card(
            "Sequence Window",
            str(WINDOW_SIZE),
            "Each prediction is generated from rolling windows of input history.",
        )
    with extra:
        render_info_card(
            "Visual Focus",
            "Clean charts",
            "Comparison views, residuals, and export-ready prediction tables.",
        )


def render_single_model_results(model_choice, y_true, y_pred, d_plot, v_plot, metrics, force_phys):
    render_section_heading(
        f"{model_choice} performance",
        "The interface below brings metrics, prediction behavior, hysteresis loops, and export tools into one focused workspace.",
    )

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_stat_card("MSE (kN²)", f"{metrics['MSE (kN²)']:.4f}", "Average squared error.", compact=True)
    with metric_cols[1]:
        render_stat_card("RMSE (kN)", f"{metrics['RMSE (kN)']:.4f}", "Primary magnitude error signal.", compact=True)
    with metric_cols[2]:
        render_stat_card("MAE (kN)", f"{metrics['MAE (kN)']:.4f}", "Median-friendly absolute deviation view.", compact=True)
    with metric_cols[3]:
        render_stat_card("MAPE (%)", f"{metrics['MAPE (%)']:.2f}", "Relative error against measured force.", compact=True)
    with metric_cols[4]:
        render_stat_card("R²", f"{metrics['R²']:.4f}", "Explained variance score.", compact=True)

    overview_tab, residual_tab, export_tab = st.tabs(
        ["Overview", "Diagnostics", "Export"]
    )

    with overview_tab:
        st.plotly_chart(
            build_prediction_overview_figure(y_true, y_pred, d_plot, v_plot, model_choice),
            use_container_width=True,
        )

    with residual_tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                build_residual_chart(y_true, y_pred, model_choice),
                use_container_width=True,
            )
        with right:
            if force_phys is not None:
                st.plotly_chart(
                    build_physics_chart(y_pred, force_phys),
                    use_container_width=True,
                )
            else:
                render_section_heading(
                    "No physics residual for this model",
                    "Physics consistency tracing is available only for the PINN model.",
                )

    with export_tab:
        export_df = pd.DataFrame(
            {
                "Actual_Force_N": y_true,
                f"{model_choice}_Pred_N": y_pred,
                "Displacement_m": d_plot,
                "Velocity_ms": v_plot,
            }
        )
        st.dataframe(export_df.head(250), use_container_width=True)
        download_csv_button(
            export_df,
            f"damper_{model_choice.lower()}_results.csv",
            "Download prediction results",
        )


def render_comparison_results(y_true, d_plot, v_plot, predictions, metrics_table, time_df, force_phys):
    best_model = metrics_table.loc["MSE (kN²)"].astype(float).idxmin()
    fastest_model = time_df.sort_values("Time (seconds)").iloc[0]["Model"]
    render_section_heading(
        "Model comparison cockpit",
        f"{best_model} currently leads on MSE. Use the tabs below to compare accuracy, runtime, hysteresis behavior, and export-ready outputs.",
    )
    render_highlight_banner(
        "Performance Lead",
        f"{best_model} is the strongest overall fit",
        f"Best MSE: {metrics_table.loc['MSE (kN²)', best_model]:.4f} kN². Fastest runtime: {fastest_model} at {time_df['Time (seconds)'].min():.4f} s.",
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_stat_card("Best MSE (kN²)", f"{metrics_table.loc['MSE (kN²)', best_model]:.4f}", "Lowest mean-squared error across all models.", compact=True)
    with metric_cols[1]:
        render_stat_card("Best R²", f"{metrics_table.loc['R²', best_model]:.4f}", "Highest explained variance in the comparison run.", compact=True)
    with metric_cols[2]:
        render_stat_card("Fastest Model", fastest_model, "Shortest inference pass for the uploaded sequence.", compact=True)
    with metric_cols[3]:
        render_stat_card("Fastest Time (s)", f"{time_df['Time (seconds)'].min():.4f}", "Measured end-to-end prediction latency.", compact=True)

    summary_tab, charts_tab, diagnostics_tab, export_tab = st.tabs(
        ["Summary", "Charts", "Diagnostics", "Export"]
    )

    with summary_tab:
        st.markdown(
            '<p class="subtle-note">Three compact model cards make the comparison easier to scan than large metric tables.</p>',
            unsafe_allow_html=True,
        )
        model_cols = st.columns(3)
        for col, model_name in zip(model_cols, ["LSTM", "PINN", "Transformer"]):
            with col:
                render_stat_card(
                    model_name,
                    f"MSE {metrics_table.loc['MSE (kN²)', model_name]:.4f} kN²",
                    f"R² {metrics_table.loc['R²', model_name]:.4f} • MAE {metrics_table.loc['MAE (kN)', model_name]:.4f} kN • Time {time_df.loc[time_df['Model'] == model_name, 'Time (seconds)'].iloc[0]:.4f} s",
                    large=True,
                )
        st.markdown(
            '<p class="subtle-note">Detailed metrics and timings are still available below for validation and export workflows.</p>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.45, 1])
        with left:
            summary_metrics_df = (
                metrics_table.T.reset_index()
                .rename(columns={"index": "Model"})
                .sort_values("MSE (kN²)")
            )
            st.dataframe(
                summary_metrics_df.style.format(
                    {
                        "MSE (kN²)": "{:.4f}",
                        "RMSE (kN)": "{:.4f}",
                        "MAE (kN)": "{:.4f}",
                        "MAPE (%)": "{:.2f}",
                        "R²": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.dataframe(
                time_df.sort_values("Time (seconds)").style.format({"Time (seconds)": "{:.6f}"}),
                use_container_width=True,
                hide_index=True,
            )

    with charts_tab:
        st.markdown(
            '<p class="subtle-note">The comparison view now prioritizes one main force trace and one paired hysteresis panel to reduce visual overload.</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_comparison_timeseries(y_true, predictions),
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.plotly_chart(
            build_hysteresis_comparison(d_plot, v_plot, y_true, predictions),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with diagnostics_tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                build_error_distribution(y_true, predictions),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.plotly_chart(
                build_difference_chart(predictions),
                use_container_width=True,
                config={"displaylogo": False},
            )
        with right:
            st.plotly_chart(
                build_interactive_hysteresis(d_plot, y_true, predictions),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.plotly_chart(
                build_physics_chart(predictions["PINN"], force_phys),
                use_container_width=True,
                config={"displaylogo": False},
            )

    with export_tab:
        export_df = pd.DataFrame(
            {
                "Actual_Force_N": y_true,
                "LSTM_Pred_N": predictions["LSTM"],
                "PINN_Pred_N": predictions["PINN"],
                "Transformer_Pred_N": predictions["Transformer"],
                "Displacement_m": d_plot,
                "Velocity_ms": v_plot,
            }
        )
        st.dataframe(export_df.head(250), use_container_width=True)
        download_csv_button(
            export_df,
            "damper_model_comparison_results.csv",
            "Download comparison results",
        )


def main():
    inject_styles()
    render_hero()

    feat_scaler, tgt_scaler = load_scalers()
    mode, model_choice, csv_file = configure_sidebar()

    if csv_file is None:
        render_empty_state()
        st.stop()

    _, x_tensor, y_true, d_plot, v_plot, summary = load_and_prepare_data(
        csv_file, feat_scaler
    )
    render_dataset_summary(summary, mode, model_choice)

    if mode == "Single Model Analysis":
        with torch.no_grad():
            if model_choice == "LSTM":
                predictions = inverse_transform_predictions(
                    load_lstm()(x_tensor).cpu().numpy(), tgt_scaler
                )
                force_phys = None
            elif model_choice == "PINN":
                pred_tensor, phys_tensor = load_pinn()(x_tensor)
                predictions = inverse_transform_predictions(
                    pred_tensor.cpu().numpy(), tgt_scaler
                )
                force_phys = phys_tensor.cpu().numpy()
            else:
                predictions = inverse_transform_predictions(
                    load_transformer()(x_tensor).cpu().numpy(), tgt_scaler
                )
                force_phys = None

        render_single_model_results(
            model_choice=model_choice,
            y_true=y_true,
            y_pred=predictions,
            d_plot=d_plot,
            v_plot=v_plot,
            metrics=compute_metrics(y_true, predictions),
            force_phys=force_phys,
        )
    else:
        with torch.no_grad():
            start = time.time()
            y_lstm = inverse_transform_predictions(
                load_lstm()(x_tensor).cpu().numpy(), tgt_scaler
            )
            lstm_time = time.time() - start

            start = time.time()
            y_pinn_tensor, force_phys_tensor = load_pinn()(x_tensor)
            y_pinn = inverse_transform_predictions(y_pinn_tensor.cpu().numpy(), tgt_scaler)
            force_phys = force_phys_tensor.cpu().numpy()
            pinn_time = time.time() - start

            start = time.time()
            y_transformer = inverse_transform_predictions(
                load_transformer()(x_tensor).cpu().numpy(), tgt_scaler
            )
            transformer_time = time.time() - start

        predictions = {
            "LSTM": y_lstm,
            "PINN": y_pinn,
            "Transformer": y_transformer,
        }
        metrics_table = pd.DataFrame(
            {name: compute_metrics(y_true, values) for name, values in predictions.items()}
        )
        time_df = pd.DataFrame(
            {
                "Model": ["LSTM", "PINN", "Transformer"],
                "Time (seconds)": [lstm_time, pinn_time, transformer_time],
            }
        )
        render_comparison_results(
            y_true=y_true,
            d_plot=d_plot,
            v_plot=v_plot,
            predictions=predictions,
            metrics_table=metrics_table,
            time_df=time_df,
            force_phys=force_phys,
        )


if __name__ == "__main__":
    main()
