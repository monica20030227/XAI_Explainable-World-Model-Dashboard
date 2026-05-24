# app.py
# ============================================================
# Explainable World Model Platform
# 完整正式版：
# 1. One-step evaluation：四模型真實 MAE / Accuracy
# 2. Multi-step rollout：v2 MLP vs v2 LSTM 真實 rollout
# 3. True Prediction Playground：從 sequence sample 跑真實 .pt inference
# 4. Real SHAP XAI：讀取離線 SHAP 結果，支援 MLP/LSTM Baseline/Temporal
#
# 執行：
#   streamlit run app.py
#
# 必要檔案建議放在同一資料夾：
#   world_model_all.csv
#   world_model_all_sequence.csv
#   world_model_mlp.pt
#   world_model_mlp_sequence.pt
#   world_model_lstm.pt
#   world_model_lstm_sequence.pt
#   world_model_scalers.pkl
#   world_model_mlp_sequence_scalers.pkl
#   world_model_lstm_scalers.pkl
#   world_model_lstm_sequence_scalers.pkl
#
# SHAP 檔案可放：
#   shap_outputs/ 內，或直接和 app.py 同層
#
# 需要套件：
#   pip install streamlit pandas numpy matplotlib torch joblib scikit-learn
# ============================================================

import os
import pickle
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Explainable World Model Platform",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Style
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1550px;
    }
    h1, h2, h3 {
        color: #172033;
        font-weight: 850 !important;
    }
    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e3a8a 100%);
        color: white;
        padding: 2.1rem 2.3rem;
        border-radius: 24px;
        margin-bottom: 1.3rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        opacity: 0.94;
        line-height: 1.75;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 900;
        color: #111827;
        margin-top: 0.5rem;
        margin-bottom: 0.9rem;
    }
    .card {
        background: white;
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 5px 16px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        min-height: 125px;
    }
    .metric-label {
        color: #6b7280;
        font-weight: 750;
        font-size: 0.95rem;
    }
    .metric-value {
        color: #111827;
        font-weight: 900;
        font-size: 1.65rem;
        margin-top: 0.25rem;
        word-break: break-word;
    }
    .metric-note {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.25rem;
        line-height: 1.55;
    }
    .box {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    .blue {
        background: #eff6ff;
        border-left: 6px solid #2563eb;
    }
    .green {
        background: #ecfdf5;
        border-left: 6px solid #059669;
    }
    .red {
        background: #fef2f2;
        border-left: 6px solid #dc2626;
    }
    .purple {
        background: #f5f3ff;
        border-left: 6px solid #7c3aed;
    }
    .yellow {
        background: #fffbeb;
        border-left: 6px solid #d97706;
    }
    .gray {
        background: #f9fafb;
        border-left: 6px solid #6b7280;
    }
    .tag {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 750;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        background: #e0f2fe;
        color: #075985;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# True Evaluation Data
# ============================================================

MODEL_NAMES = {
    "MLP v1": "MLP (Baseline)",
    "LSTM v1": "LSTM (Baseline)",
    "MLP v2": "MLP (Temporal)",
    "LSTM v2": "LSTM (Temporal)",
}

MODEL_DISPLAY_COLS = MODEL_NAMES.copy()

SHAP_PREFIX = {
    "MLP v1": "mlp_v1",
    "MLP v2": "mlp_v2",
    "LSTM v1": "lstm_v1",
    "LSTM v2": "lstm_v2",
}

ONE_STEP_DATA = pd.DataFrame(
    [
        ["speed_next", "MAE", 0.2660, 0.2991, 0.2331, 0.2985],
        ["acceleration_next", "MAE", 0.0683, 0.0815, 0.0577, 0.0688],
        ["dist_tls_next", "MAE", 13.8117, 14.6604, 11.7079, 12.6626],
        ["tls_color_next", "MAE", 0.1159, 0.0808, 0.0955, 0.0706],
        ["leader_gap_next", "MAE", 3.6208, 2.8786, 3.1692, 3.0649],
        ["leader_speed_next", "MAE", 0.5360, 0.5389, 0.4646, 0.4674],
        ["waiting_time_next", "MAE", 0.0764, 0.0553, 0.0626, 0.0171],
        ["time_loss_next", "MAE", 2.9015, 3.5633, 2.3804, 3.2993],
        ["is_stopped_next", "Accuracy", 0.9971, 0.9993, 0.9980, 0.9991],
        ["reward", "MAE", 0.1248, 0.1284, 0.1204, 0.1233],
        ["done", "Accuracy", 0.9923, 0.9921, 0.9932, 0.9895],
    ],
    columns=["Target", "Metric", "MLP v1", "LSTM v1", "MLP v2", "LSTM v2"],
)

MLP_V2_ROLLOUT = pd.DataFrame(
    [
        [10, 9344, 0.3270228269432147, 0.6388866115421657, 56.99679417907769, 0.43362657742688687, 5.28801415822342, 1.2828115467394035, 0.30395830347406444, 7.564717987473593, 0.9717144691780822, 0.43702597236495855, 1.0],
        [20, 8327, 0.575036614527718, 0.7277005621202107, 123.9110671146485, 0.7374344502033028, 7.7875793450879325, 2.3407508962635886, 0.4187492114646804, 14.78489240925165, 0.9551699291461511, 0.5171161613654465, 1.0],
        [50, 3944, 72.57588213081475, 42.2660683279757, 7615.231144305236, 45.23618411129129, 246.56843393464877, 136.08239184780447, 31.46985818833058, 3167.665673192409, 0.8976166328600406, 17.631230822024076, 0.9999087221095335],
    ],
    columns=[
        "Horizon", "total_start_points", "speed_next_mae", "acceleration_next_mae",
        "dist_tls_next_mae", "tls_color_next_mae", "leader_gap_next_mae",
        "leader_speed_next_mae", "waiting_time_next_mae", "time_loss_next_mae",
        "is_stopped_next_accuracy", "reward_mae", "done_accuracy",
    ],
)

LSTM_V2_ROLLOUT = pd.DataFrame(
    [
        [10, 8327, 0.3293344835029111, 0.2878009403814953, 25.51720164619839, 0.4123343314698549, 4.044872221887445, 0.9104308173495045, 0.415612166933791, 3.076514296150347, 0.9724510628077339, 0.20985298554912152, 0.9999639726191906],
        [20, 6644, 0.6540198458166797, 0.3779373034041335, 56.66452220383076, 0.7571006913837061, 9.41115226389784, 1.781314018044476, 0.7797634023261023, 5.3105105597683355, 0.9442203491872366, 0.34954629130816756, 0.9929485249849488],
        [50, 3238, 0.5935301567183706, 0.4672844906435296, 102.04834702125423, 0.8060328254553663, 11.773126053084356, 3.464049353241179, 1.2258972088320677, 10.905281078904354, 0.9104694255713404, 0.4426170767893669, 0.9989005558987029],
    ],
    columns=MLP_V2_ROLLOUT.columns,
)

TARGETS = [
    "speed_next",
    "acceleration_next",
    "dist_tls_next",
    "tls_color_next",
    "leader_gap_next",
    "leader_speed_next",
    "waiting_time_next",
    "time_loss_next",
    "is_stopped_next",
    "reward",
    "done",
]

TARGET_LABELS = {
    "speed_next": "下一步車速",
    "acceleration_next": "下一步加速度",
    "dist_tls_next": "下一步距離號誌",
    "tls_color_next": "下一步號誌狀態",
    "leader_gap_next": "下一步前車距離",
    "leader_speed_next": "下一步前車速度",
    "waiting_time_next": "下一步等待時間",
    "time_loss_next": "下一步時間損失",
    "is_stopped_next": "下一步是否停車",
    "reward": "Reward",
    "done": "Episode 結束判斷",
}

FEATURE_ANALYSIS = {
    "target_speed": "目標速度代表控制意圖，若排名靠前，表示模型會根據控制目標推估下一步狀態。",
    "speed_t": "目前車速是最核心的動態特徵，合理影響速度、加速度、等待、time loss 與 reward。",
    "acceleration_t": "目前加速度反映瞬間運動狀態，通常輔助速度與 time loss 預測。",
    "dist_tls_t": "與號誌距離是路口情境的重要資訊，對距離、號誌與停等預測具有合理影響。",
    "tls_color_t": "目前號誌狀態影響停等、號誌轉移與控制策略，是交通決策的關鍵變數。",
    "leader_gap_t": "前車距離影響跟車安全與車流互動，對 leader_gap_next 合理重要。",
    "leader_speed_t": "前車速度反映前方車流動態，對 leader_speed_next 與跟車情境有合理影響。",
    "waiting_time_t": "目前等待時間具有時間延續性，對 waiting_time_next 合理重要。",
    "time_loss_t": "目前時間損失具累積性，若排名靠前代表模型學到交通延遲的 temporal persistence。",
    "is_stopped_t": "是否停車會影響等待時間與下一步停車狀態，是停等判斷的核心特徵。",
    "action": "控制動作影響下一步狀態；若排名較低，表示模型更依賴交通物理狀態而非單一動作標籤。",
    "vehicle_type_ICE": "ICE 車種對 reward 較重要，可能反映燃油與 CO₂ 成本差異。",
    "vehicle_type_EV": "EV 車種對 reward 較重要，可能反映能源消耗與排放機制差異。",
    "vehicle_type_HEV": "HEV 車種對 reward 有影響，可能反映混合動力能耗機制。",
    "source_policy_DQN": "若 source_policy SHAP 較低，代表模型不是主要記住資料來源 policy，而是在學交通動態。",
    "source_policy_DDQN": "若 source_policy SHAP 較低，代表模型不是主要記住資料來源 policy，而是在學交通動態。",
    "source_policy_DDPG": "若 source_policy SHAP 較低，代表模型不是主要記住資料來源 policy，而是在學交通動態。",
}

TARGET_ANALYSIS = {
    "speed_next": {
        "title": "Speed Prediction",
        "text": "speed_next 通常主要依賴 target_speed 與 speed_t，代表模型根據控制目標與當前速度推估下一步速度。",
        "expected": ["target_speed", "speed_t", "acceleration_t", "action"],
    },
    "acceleration_next": {
        "title": "Acceleration Prediction",
        "text": "acceleration_next 主要受 speed_t 與 target_speed 影響，表示模型學到加速度與速度控制目標的關係。",
        "expected": ["speed_t", "target_speed", "action", "leader_gap_t"],
    },
    "dist_tls_next": {
        "title": "Distance-to-Signal Prediction",
        "text": "dist_tls_next 由 dist_tls_t 主導是合理的，下一步距離主要由目前距離與車輛運動狀態決定。",
        "expected": ["dist_tls_t", "speed_t", "tls_color_t"],
    },
    "tls_color_next": {
        "title": "Traffic Signal Prediction",
        "text": "tls_color_next 主要依賴 tls_color_t，代表號誌狀態轉移具有延續性與規則性。",
        "expected": ["tls_color_t", "dist_tls_t", "time_loss_t"],
    },
    "leader_gap_next": {
        "title": "Leader Gap Prediction",
        "text": "leader_gap_next 主要依賴 leader_gap_t，並受到速度與前車狀態影響，符合跟車距離邏輯。",
        "expected": ["leader_gap_t", "speed_t", "leader_speed_t", "target_speed"],
    },
    "leader_speed_next": {
        "title": "Leader Speed Prediction",
        "text": "leader_speed_next 主要依賴 leader_speed_t，代表模型捕捉前車速度延續性。",
        "expected": ["leader_speed_t", "leader_gap_t", "speed_t"],
    },
    "waiting_time_next": {
        "title": "Waiting Time Prediction",
        "text": "waiting_time_next 依賴 is_stopped_t、waiting_time_t 與 tls_color_t，顯示模型學到停等狀態的累積與號誌影響。",
        "expected": ["is_stopped_t", "waiting_time_t", "tls_color_t", "speed_t"],
    },
    "time_loss_next": {
        "title": "Time Loss Prediction",
        "text": "time_loss_next 主要依賴 time_loss_t，代表時間損失具有強烈累積性。",
        "expected": ["time_loss_t", "speed_t", "target_speed"],
    },
    "is_stopped_next": {
        "title": "Stop State Prediction",
        "text": "is_stopped_next 受 is_stopped_t、speed_t、dist_tls_t 與 tls_color_t 影響，符合停等判斷邏輯。",
        "expected": ["is_stopped_t", "speed_t", "dist_tls_t", "tls_color_t"],
    },
    "reward": {
        "title": "Reward Prediction",
        "text": "reward 除了受速度與目標速度影響，也可能受到 vehicle_type 影響，表示模型捕捉不同車種在能耗或 reward function 中的差異。",
        "expected": ["speed_t", "target_speed", "vehicle_type_ICE", "vehicle_type_EV", "vehicle_type_HEV"],
    },
    "done": {
        "title": "Done Prediction",
        "text": "done 與 time_loss_t、speed_t、dist_tls_t 或 target_speed 等狀態有關，反映 episode 結束判斷與交通進程之間的關聯。",
        "expected": ["time_loss_t", "speed_t", "target_speed", "dist_tls_t"],
    },
}


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(".")
SHAP_DIR = BASE_DIR / "shap_outputs"
SEARCH_DIRS = [SHAP_DIR, BASE_DIR]

DATA_PATHS = {
    "v1": BASE_DIR / "world_model_all.csv",
    "v2": BASE_DIR / "world_model_all_sequence.csv",
}

MODEL_PATHS = {
    "MLP v1": BASE_DIR / "world_model_mlp.pt",
    "MLP v2": BASE_DIR / "world_model_mlp_sequence.pt",
    "LSTM v1": BASE_DIR / "world_model_lstm.pt",
    "LSTM v2": BASE_DIR / "world_model_lstm_sequence.pt",
}

SCALER_PATHS = {
    "MLP v1": BASE_DIR / "world_model_scalers.pkl",
    "MLP v2": BASE_DIR / "world_model_mlp_sequence_scalers.pkl",
    "LSTM v1": BASE_DIR / "world_model_lstm_scalers.pkl",
    "LSTM v2": BASE_DIR / "world_model_lstm_sequence_scalers.pkl",
}


# ============================================================
# Model Definitions
# ============================================================

if TORCH_AVAILABLE:
    class MLPWorldModel(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    class LSTMWorldModel(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)


# ============================================================
# General Helpers
# ============================================================

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🚦 Explainable World Model Platform</div>
            <div class="hero-subtitle">
                智慧交通號誌路口車速控制之 world model prediction、multi-step rollout 與 SHAP explainability dashboard。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text, color="blue"):
    st.markdown(f'<div class="box {color}">{text}</div>', unsafe_allow_html=True)


def show_tags(items):
    html = "".join([f'<span class="tag">{x}</span>' for x in items])
    st.markdown(html, unsafe_allow_html=True)


def safe_format_df(df, digits=4):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df.style.format({c: f"{{:.{digits}f}}" for c in numeric_cols})


def display_model_name(model_key: str):
    return MODEL_NAMES.get(model_key, model_key)


def display_model_columns(df: pd.DataFrame):
    display_df = df.copy()
    display_df = display_df.rename(columns=MODEL_DISPLAY_COLS)
    if "Best Model" in display_df.columns:
        display_df["Best Model"] = display_df["Best Model"].map(lambda x: MODEL_NAMES.get(x, x))
    return display_df


def find_existing_file(filename: str):
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


# ============================================================
# Plot Helpers
# ============================================================

def plot_barh(df, feature_col, value_col, title, xlabel="Mean |SHAP value|", top_n=17):
    plot_df = df[[feature_col, value_col]].copy()
    plot_df = plot_df.sort_values(value_col, ascending=False).head(top_n)
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df[feature_col], plot_df[value_col])
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_one_step_target(target):
    row = ONE_STEP_DATA[ONE_STEP_DATA["Target"] == target].iloc[0]
    models = ["MLP v1", "LSTM v1", "MLP v2", "LSTM v2"]
    display_labels = [display_model_name(m) for m in models]
    values = [row[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(display_labels, values)
    ax.set_title(f"{target} ({row['Metric']})", fontsize=15, fontweight="bold")
    ax.set_ylabel(row["Metric"])
    ax.grid(axis="y", alpha=0.25)

    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_rollout(metric_col, title, ylabel="MAE / Accuracy"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(MLP_V2_ROLLOUT["Horizon"], MLP_V2_ROLLOUT[metric_col], marker="o", linewidth=2, label="MLP (Temporal)")
    ax.plot(LSTM_V2_ROLLOUT["Horizon"], LSTM_V2_ROLLOUT[metric_col], marker="o", linewidth=2, label="LSTM (Temporal)")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Rollout Horizon")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_model_compare(df_a, df_b, label_a, label_b, value_col, title, top_n=12):
    merged = pd.merge(
        df_a[["feature", value_col]],
        df_b[["feature", value_col]],
        on="feature",
        how="outer",
        suffixes=(f"_{label_a}", f"_{label_b}"),
    ).fillna(0)

    col_a = f"{value_col}_{label_a}"
    col_b = f"{value_col}_{label_b}"

    merged["max_value"] = merged[[col_a, col_b]].max(axis=1)
    merged["delta_b_minus_a"] = merged[col_b] - merged[col_a]

    plot_df = merged.sort_values("max_value", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("max_value", ascending=True)

    y = np.arange(len(plot_df))
    height = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - height / 2, plot_df[col_a], height, label=label_a)
    ax.barh(y + height / 2, plot_df[col_b], height, label=label_b)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    plt.tight_layout()

    return fig, merged.sort_values("max_value", ascending=False)


# ============================================================
# Evaluation Helpers
# ============================================================

def best_model_summary():
    rows = []
    for _, row in ONE_STEP_DATA.iterrows():
        values = row[["MLP v1", "LSTM v1", "MLP v2", "LSTM v2"]]
        if row["Metric"] == "Accuracy":
            best = values.astype(float).idxmax()
        else:
            best = values.astype(float).idxmin()
        rows.append([row["Target"], row["Metric"], best, row[best]])
    return pd.DataFrame(rows, columns=["Target", "Metric", "Best Model", "Best Value"])


# ============================================================
# SHAP Helpers
# ============================================================

@st.cache_data
def load_shap_csv(model_key: str, target: str = "global"):
    prefix = SHAP_PREFIX[model_key]
    if target == "global":
        filename = f"{prefix}_global_feature_importance.csv"
    else:
        filename = f"{prefix}_{target}_feature_importance.csv"

    path = find_existing_file(filename)
    if path is None:
        return None, filename

    return pd.read_csv(path), filename


@st.cache_data
def load_shap_pkl(model_key: str):
    prefix = SHAP_PREFIX[model_key]
    filename = f"{prefix}_shap_values.pkl"
    path = find_existing_file(filename)
    if path is None:
        return None, filename

    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj, filename


def detect_shap_value_column(df):
    if "mean_abs_shap_all_targets" in df.columns:
        return "mean_abs_shap_all_targets"
    if "mean_abs_shap" in df.columns:
        return "mean_abs_shap"
    candidates = [c for c in df.columns if c != "feature" and pd.api.types.is_numeric_dtype(df[c])]
    if candidates:
        return candidates[0]
    return None


def analyze_top_features(df, value_col, target=None, top_k=6):
    top = df.sort_values(value_col, ascending=False).head(top_k)
    features = top["feature"].tolist()

    lines = []

    if target and target in TARGET_ANALYSIS:
        lines.append(f"<b>{TARGET_ANALYSIS[target]['title']}：</b>{TARGET_ANALYSIS[target]['text']}")

    lines.append("<b>Top SHAP features：</b>")
    for _, row in top.iterrows():
        feat = row["feature"]
        value = row[value_col]
        desc = FEATURE_ANALYSIS.get(feat, "此特徵對模型輸出具有影響。")
        lines.append(f"- <b>{feat}</b>（Mean |SHAP| = {value:.4f}）：{desc}")

    if target and target in TARGET_ANALYSIS:
        expected = TARGET_ANALYSIS[target]["expected"]
        hit = [f for f in features if f in expected]
        if hit:
            lines.append(f"<br><b>合理性檢查：</b>Top features 中包含 {', '.join(hit)}，與該 target 的交通直覺相符。")
        else:
            lines.append("<br><b>合理性檢查：</b>Top features 未明顯包含預期關鍵變數，建議進一步檢查資料分布。")

    return "<br>".join(lines)


def comparison_text(df_a, df_b, label_a, label_b, value_col):
    merged = pd.merge(
        df_a[["feature", value_col]],
        df_b[["feature", value_col]],
        on="feature",
        how="outer",
        suffixes=(f"_{label_a}", f"_{label_b}"),
    ).fillna(0)

    col_a = f"{value_col}_{label_a}"
    col_b = f"{value_col}_{label_b}"
    merged["delta"] = merged[col_b] - merged[col_a]

    up = merged.sort_values("delta", ascending=False).head(3)
    down = merged.sort_values("delta", ascending=True).head(3)

    text = f"<b>{label_a} vs {label_b} 差異解讀：</b><br>"
    text += f"{label_b} 相較 {label_a} 權重增加較明顯的特徵："
    text += ", ".join([f"{r['feature']} ({r['delta']:.4f})" for _, r in up.iterrows()])
    text += f"<br>{label_b} 相較 {label_a} 權重下降較明顯的特徵："
    text += ", ".join([f"{r['feature']} ({r['delta']:.4f})" for _, r in down.iterrows()])
    text += "<br><br>若 LSTM 對 time_loss_t、waiting_time_t、tls_color_t 或 dist_tls_t 的依賴提高，可解讀為序列模型更重視時間延續性與交通狀態變化。"
    return text


def available_shap_models():
    models = []
    for m in MODEL_NAMES:
        df, _ = load_shap_csv(m, "global")
        if df is not None:
            models.append(m)
    return models


# ============================================================
# Inference Helpers
# ============================================================

def prepare_input_dataframe(raw_df: pd.DataFrame, input_columns: list):
    df = raw_df.copy()

    if "vehicle_type" in df.columns:
        vehicle_clean = df["vehicle_type"].astype(str).str.strip().str.upper()
        for v in ["EV", "HEV", "ICE"]:
            col = f"vehicle_type_{v}"
            if col in input_columns:
                df[col] = (vehicle_clean == v).astype(int)

    if "source_policy" in df.columns:
        policy_clean = df["source_policy"].astype(str).str.strip().str.upper()
        for p in ["DDPG", "DDQN", "DQN"]:
            col = f"source_policy_{p}"
            if col in input_columns:
                df[col] = (policy_clean == p).astype(int)

    for col in input_columns:
        if col not in df.columns:
            df[col] = 0

    X_df = df[input_columns].copy()

    for col in X_df.columns:
        if X_df[col].dtype == bool:
            X_df[col] = X_df[col].astype(int)
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X_df = X_df.fillna(0)

    return X_df


@st.cache_resource
def load_scaler_bundle(model_key):
    path = SCALER_PATHS.get(model_key)
    if path is None or not path.exists():
        return None, f"找不到 scaler：{path}"
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)


def infer_lstm_hyperparams_from_state_dict(state_dict, default_hidden=128, default_layers=2):
    hidden_dim = default_hidden
    num_layers = default_layers

    for key, value in state_dict.items():
        if key.endswith("weight_ih_l0") or key == "lstm.weight_ih_l0":
            hidden_dim = int(value.shape[0] // 4)
            break

    layer_ids = []
    for key in state_dict.keys():
        if "lstm.weight_ih_l" in key:
            try:
                layer_str = key.split("lstm.weight_ih_l")[-1]
                layer_id = int(layer_str.split(".")[0])
                layer_ids.append(layer_id)
            except Exception:
                pass

    if layer_ids:
        num_layers = max(layer_ids) + 1

    return hidden_dim, num_layers


@st.cache_resource
def load_torch_model(model_key, input_dim, output_dim):
    if not TORCH_AVAILABLE:
        return None, "目前環境沒有 torch，無法載入 .pt 模型。"

    model_path = MODEL_PATHS.get(model_key)
    if model_path is None or not model_path.exists():
        return None, f"找不到模型檔案：{model_path}"

    device = torch.device("cpu")

    try:
        loaded = torch.load(model_path, map_location=device)

        if isinstance(loaded, nn.Module):
            loaded.to(device)
            loaded.eval()
            return loaded, None

        if isinstance(loaded, dict):
            state_dict = loaded["model_state_dict"] if "model_state_dict" in loaded else loaded

            if "MLP" in model_key:
                model = MLPWorldModel(input_dim=input_dim, output_dim=output_dim)
            else:
                hidden_dim, num_layers = infer_lstm_hyperparams_from_state_dict(state_dict)
                model = LSTMWorldModel(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                )

            model.load_state_dict(state_dict)
            model.eval()
            return model, None

        return None, f"不支援的模型格式：{type(loaded)}"

    except Exception as e:
        return None, str(e)


@st.cache_data
def load_sequence_data():
    path = DATA_PATHS["v2"]
    if not path.exists():
        return None, f"找不到資料檔案：{path}"
    try:
        df = pd.read_csv(path)
        if "sequence_id" not in df.columns:
            return None, "world_model_all_sequence.csv 缺少 sequence_id 欄位。"
        return df, None
    except Exception as e:
        return None, str(e)


def run_true_inference(model_key, selected_rows):
    scalers, scaler_err = load_scaler_bundle(model_key)
    if scaler_err:
        return None, None, scaler_err

    required = ["x_scaler", "y_scaler", "input_columns", "target_columns"]
    missing = [k for k in required if k not in scalers]
    if missing:
        return None, None, f"Scaler 缺少欄位：{missing}"

    input_columns = list(scalers["input_columns"])
    target_columns = list(scalers["target_columns"])
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    model, model_err = load_torch_model(model_key, len(input_columns), len(target_columns))
    if model_err:
        return None, None, model_err

    X_df = prepare_input_dataframe(selected_rows, input_columns)
    X_raw = X_df.values.astype(np.float32)
    X_scaled = x_scaler.transform(X_raw).astype(np.float32)

    if not TORCH_AVAILABLE:
        return None, None, "torch 不可用。"

    with torch.no_grad():
        if "LSTM" in model_key:
            seq_len = int(scalers.get("seq_len", 10))
            if len(X_scaled) < seq_len:
                pad_count = seq_len - len(X_scaled)
                pad = np.repeat(X_scaled[:1], pad_count, axis=0)
                X_seq = np.vstack([pad, X_scaled])
            else:
                X_seq = X_scaled[-seq_len:]

            x_tensor = torch.tensor(X_seq[np.newaxis, :, :], dtype=torch.float32)
            pred_scaled = model(x_tensor).detach().cpu().numpy()
        else:
            x_tensor = torch.tensor(X_scaled[-1:, :], dtype=torch.float32)
            pred_scaled = model(x_tensor).detach().cpu().numpy()

    pred = y_scaler.inverse_transform(pred_scaled)[0]

    pred_df = pd.DataFrame({
        "Prediction Target": target_columns,
        "Predicted Value": pred,
    })

    actual_cols = [c for c in target_columns if c in selected_rows.columns]
    if actual_cols:
        actual = selected_rows.iloc[-1][actual_cols].values.astype(float)
        actual_df = pd.DataFrame({
            "Prediction Target": actual_cols,
            "Actual Value": actual,
        })
        compare_df = pd.merge(pred_df, actual_df, on="Prediction Target", how="left")
        compare_df["Absolute Error"] = (compare_df["Predicted Value"] - compare_df["Actual Value"]).abs()
    else:
        compare_df = pred_df

    input_display = X_df.iloc[-1:].T.reset_index()
    input_display.columns = ["Input Feature", "Value"]

    return compare_df, input_display, None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🚦 Navigation")
    page = st.radio(
        "選擇頁面",
        [
            "1. Project Overview",
            "2. Dataset Design",
            "3. One-step Evaluation",
            "4. Multi-step Rollout",
            "5. True Prediction Playground",
            "6. Real SHAP XAI",
            "7. Research Findings",
        ],
    )

    st.divider()

    selected_model = st.selectbox(
        "模型選擇",
        list(MODEL_NAMES.keys()),
        format_func=lambda x: MODEL_NAMES[x],
    )

    st.caption("Baseline = Static dataset；Temporal = Sequence / episode context")
    st.caption("SHAP 為離線真實結果，不在 app 即時計算。")


render_hero()


# ============================================================
# Pages
# ============================================================

if page == "1. Project Overview":
    st.markdown('<div class="section-title">研究平台定位</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("World Model Task", "sₜ + aₜ → sₜ₊₁", "根據目前交通狀態與控制動作預測下一步")
    with c2:
        metric_card("Models", "4", "MLP / LSTM × Baseline / Temporal")
    with c3:
        metric_card("Evaluation", "MAE / Acc")
    with c4:
        metric_card("XAI", "Offline SHAP", "MLP/LSTM 皆支援")

    info_box(
        """
        本平台將智慧交通車速控制問題轉換為 world model prediction task。
        系統整合模型效能比較、multi-step rollout、真實 .pt inference，
        以及離線真實 SHAP explainability analysis。
        """,
        "blue",
    )

    st.markdown("### 模型命名與角色")
    model_df = pd.DataFrame(
        [
            [ "MLP (Baseline)", "作為 static baseline"],
            [ "LSTM (Baseline)", "用於觀察序列記憶效果"],
            ["MLP (Temporal)", "使用 temporal data"],
            [ "LSTM (Temporal)", "使用 temporal data"],
        ],
        columns=[ "模型名稱", "說明"],
    )
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("### 分析模組")
    show_tags([
        "One-step Evaluation",
        "Multi-step Rollout",
        "True .pt Inference",
        "MLP SHAP",
        "LSTM SHAP",
        "Target-level XAI",
        "Model Comparison",
        "Dreamer Future Work",
    ])


elif page == "2. Dataset Design":
    st.markdown('<div class="section-title">Baseline / Temporal Dataset Feature Design</div>', unsafe_allow_html=True)

    info_box(
        """
        <b>Baseline</b> 是 static / baseline dataset，主要用於單一步交通狀態轉移預測。<br>
        <b>Temporal</b> 加入 temporal context，例如 sequence_id、episode、step，
        使資料更適合 sequence-aware rollout 與 long-horizon prediction。
        """,
        "green",
    )

    feature_df = pd.DataFrame(
        [
            ["Metadata", "vehicle_type, source_policy", "Baseline / Temporal 共用"],
            ["Current State", "speed_t, acceleration_t, dist_tls_t, tls_color_t, leader_gap_t, leader_speed_t, waiting_time_t, time_loss_t, is_stopped_t", "Baseline / Temporal 共用"],
            ["Action / Control", "action, target_speed", "Baseline / Temporal 共用"],
            ["Learning Signal", "reward, done", "Baseline / Temporal 共用"],
            ["Prediction Targets", "speed_next, acceleration_next, dist_tls_next, tls_color_next, leader_gap_next, leader_speed_next, waiting_time_next, time_loss_next, is_stopped_next, reward, done", "Baseline / Temporal 共用"],
            ["Temporal Index", "sequence_id, episode, step", "Temporal 新增"],
        ],
        columns=["欄位類型", "欄位名稱", "版本"],
    )

    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        info_box(
            """
            <b>Baseline</b><br>
            - 著重單一步 state transition<br>
            - 適合作為 baseline 與 SHAP 參考<br>
            - 較難表達長期 temporal dependency
            """,
            "blue",
        )
    with col2:
        info_box(
            """
            <b>Temporal</b><br>
            - 保留 sequence 與 episode 資訊<br>
            - 更適合 multi-step rollout<br>
            - 可觀察 temporal context 是否改變 feature reliance
            """,
            "purple",
        )


elif page == "3. One-step Evaluation":
    st.markdown('<div class="section-title">One-step Evaluation：四模型真實數據比較</div>', unsafe_allow_html=True)

    info_box(
        """
        此頁只顯示正式 evaluation log。連續變數使用 <b>MAE</b>，
        狀態 / 分類變數使用 <b>Accuracy</b>。
        """,
        "blue",
    )

    st.markdown("### Complete Evaluation Table")
    st.dataframe(safe_format_df(display_model_columns(ONE_STEP_DATA)), use_container_width=True, hide_index=True)

    st.markdown("### Best Model by Target")
    best_df = best_model_summary()
    st.dataframe(safe_format_df(display_model_columns(best_df)), use_container_width=True, hide_index=True)

    st.markdown("### Target Visualization")
    selected_targets = st.multiselect(
        "選擇要視覺化的 target",
        ONE_STEP_DATA["Target"].tolist(),
        default=["speed_next", "dist_tls_next", "waiting_time_next", "is_stopped_next"],
    )

    for target in selected_targets:
        st.pyplot(plot_one_step_target(target), use_container_width=True)

    info_box(
        """
        <b>觀察：</b><br>
        MLP (Temporal) 在 speed、acceleration、dist_tls、leader_speed、time_loss、reward 等多數連續變數上表現較佳；
        LSTM (Temporal) 在 tls_color 與 waiting_time 上表現突出；
        LSTM (Baseline) 在 leader_gap 與 is_stopped accuracy 上具有優勢。
        """,
        "green",
    )


elif page == "4. Multi-step Rollout":
    st.markdown('<div class="section-title">Multi-step Rollout：MLP (Temporal) vs LSTM (Temporal)</div>', unsafe_allow_html=True)

    info_box(
        """
        此頁只使用 MLP (Temporal) 與 LSTM (Temporal) multi-step rollout 真實數據。
        研究重點是新版 temporal model 的 long-horizon stability。
        """,
        "blue",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Rollout Horizons", "10 / 20 / 50", "短期、中期、長期預測")
    with c2:
        metric_card("MLP (Temporal) 50-step speed MAE", f"{MLP_V2_ROLLOUT.loc[2, 'speed_next_mae']:.4f}", "長期 rollout 明顯發散")
    with c3:
        metric_card("LSTM (Temporal) 50-step speed MAE", f"{LSTM_V2_ROLLOUT.loc[2, 'speed_next_mae']:.4f}", "長期穩定性較佳")

    compare_df = pd.DataFrame({
        "Horizon": MLP_V2_ROLLOUT["Horizon"],
        "MLP (Temporal) speed MAE": MLP_V2_ROLLOUT["speed_next_mae"],
        "LSTM (Temporal) speed MAE": LSTM_V2_ROLLOUT["speed_next_mae"],
        "MLP (Temporal) dist_tls MAE": MLP_V2_ROLLOUT["dist_tls_next_mae"],
        "LSTM (Temporal) dist_tls MAE": LSTM_V2_ROLLOUT["dist_tls_next_mae"],
        "MLP (Temporal) time_loss MAE": MLP_V2_ROLLOUT["time_loss_next_mae"],
        "LSTM (Temporal) time_loss MAE": LSTM_V2_ROLLOUT["time_loss_next_mae"],
        "MLP (Temporal) reward MAE": MLP_V2_ROLLOUT["reward_mae"],
        "LSTM (Temporal) reward MAE": LSTM_V2_ROLLOUT["reward_mae"],
    })

    st.markdown("### Rollout Summary")
    st.dataframe(safe_format_df(compare_df), use_container_width=True, hide_index=True)

    with st.expander("查看完整 MLP (Temporal) rollout 指標"):
        st.dataframe(safe_format_df(MLP_V2_ROLLOUT), use_container_width=True, hide_index=True)

    with st.expander("查看完整 LSTM (Temporal) rollout 指標"):
        st.dataframe(safe_format_df(LSTM_V2_ROLLOUT), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_rollout("speed_next_mae", "V2 Rollout: speed_next MAE", "MAE"), use_container_width=True)
        st.pyplot(plot_rollout("time_loss_next_mae", "V2 Rollout: time_loss_next MAE", "MAE"), use_container_width=True)
    with col2:
        st.pyplot(plot_rollout("dist_tls_next_mae", "V2 Rollout: dist_tls_next MAE", "MAE"), use_container_width=True)
        st.pyplot(plot_rollout("reward_mae", "V2 Rollout: reward MAE", "MAE"), use_container_width=True)

    info_box(
        """
        <b>核心發現：</b><br>
        在 multi-step rollout 中，MLP (Temporal) 於 50-step 發生明顯 error accumulation，
        而 LSTM (Temporal) 能維持較穩定的動態預測。這表示 temporal memory 對 world model 的長期 imagination 很重要。
        """,
        "red",
    )


elif page == "5. True Prediction Playground":
    st.markdown('<div class="section-title">True Prediction Playground：從真實 sequence sample 跑 .pt inference</div>', unsafe_allow_html=True)

    info_box(
        """
        此頁不計算 SHAP，只負責真實 world model prediction。
        SHAP 屬於 offline explainability analysis，請到 Real SHAP XAI 頁查看。
        """,
        "yellow",
    )

    df_seq, err = load_sequence_data()
    if err:
        st.error(err)
    else:
        st.markdown(f"### Selected Model: {MODEL_NAMES[selected_model]}")

        sequence_ids = (
            df_seq["sequence_id"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            seq_id = st.selectbox("選擇 sequence_id", sequence_ids, index=0)
        with col2:
            seq_rows_all = df_seq[df_seq["sequence_id"].astype(str) == str(seq_id)].copy()
            st.metric("Sequence Rows", len(seq_rows_all))
        with col3:
            max_n = min(30, max(1, len(seq_rows_all)))
            default_n = min(10, max_n)
            use_last_n = st.slider("使用最後 N 筆做推論", 1, max_n, default_n)

        selected_rows = seq_rows_all.tail(use_last_n).copy()

        with st.expander("查看使用的 sequence sample"):
            st.dataframe(selected_rows, use_container_width=True)

        pred_df, input_df, infer_err = run_true_inference(selected_model, selected_rows)

        if infer_err:
            st.error(infer_err)
            st.info("請確認 .pt、scaler.pkl、world_model_all_sequence.csv 都與 app.py 在同一個資料夾。")
        else:
            st.markdown("### Prediction vs Actual")
            st.dataframe(safe_format_df(pred_df), use_container_width=True, hide_index=True)

            st.markdown("### Model Input Features")
            st.dataframe(input_df, use_container_width=True, hide_index=True)

            info_box(
                """
                <b>解讀方式：</b><br>
                MLP 使用最後一筆 state；LSTM 使用最後 seq_len 筆 sequence。
                此頁是 prediction module，不混合 SHAP 計算。
                """,
                "blue",
            )


elif page == "6. Real SHAP XAI":
    st.markdown('<div class="section-title">Real SHAP XAI：四模型離線真實 SHAP 分析</div>', unsafe_allow_html=True)

    info_box(
        """
        本頁讀取本地端完成的 SHAP CSV / PKL。
        MLP 與 LSTM 都使用離線 sample-based approximate SHAP。
        """,
        "green",
    )

    shap_models = available_shap_models()
    if not shap_models:
        st.error("目前找不到任何 SHAP CSV。請確認 shap_outputs/ 中有 *_global_feature_importance.csv。")
        st.stop()

    shap_mode = st.radio(
        "選擇 XAI 分析模式",
        [
            "Global SHAP",
            "Target-level SHAP",
            "Model Comparison",
            "Raw SHAP Summary",
            "XAI Findings",
        ],
        horizontal=True,
    )

    if shap_mode == "Global SHAP":
        shap_model = st.selectbox("選擇模型", shap_models, format_func=lambda x: MODEL_NAMES[x])
        global_df, filename = load_shap_csv(shap_model, "global")

        if global_df is None:
            st.error(f"找不到 {filename}")
        else:
            value_col = detect_shap_value_column(global_df)

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("SHAP Source", filename, "離線真實 SHAP")
            with c2:
                metric_card("Features", len(global_df), "Input features")
            with c3:
                metric_card("Top Feature", global_df.sort_values(value_col, ascending=False).iloc[0]["feature"], "Global importance 最大")

            col1, col2 = st.columns([1.25, 1])
            with col1:
                st.pyplot(
                    plot_barh(
                        global_df,
                        "feature",
                        value_col,
                        f"{display_model_name(shap_model)} Global SHAP Importance - All Targets",
                    ),
                    use_container_width=True,
                )
            with col2:
                st.markdown("### Feature Ranking")
                st.dataframe(safe_format_df(global_df), use_container_width=True, hide_index=True)

            info_box(analyze_top_features(global_df, value_col, target=None, top_k=7), "blue")

            info_box(
                """
                <b>總體解讀：</b><br>
                若 target_speed、speed_t、dist_tls_t、time_loss_t、tls_color_t 排名靠前，
                表示模型主要依賴交通狀態與控制目標。若 source_policy 的 SHAP 值偏低，
                代表模型不是主要記住資料來源 policy，而是在學交通動態。
                """,
                "green",
            )

    elif shap_mode == "Target-level SHAP":
        col_a, col_b = st.columns(2)
        with col_a:
            shap_model = st.selectbox("選擇模型", shap_models, format_func=lambda x: MODEL_NAMES[x])
        with col_b:
            target = st.selectbox(
                "選擇 prediction target",
                TARGETS,
                format_func=lambda x: f"{x}（{TARGET_LABELS.get(x, x)}）",
            )

        target_df, filename = load_shap_csv(shap_model, target)

        if target_df is None:
            st.error(f"找不到 {filename}。請確認 target-level SHAP csv 已放在 shap_outputs/ 或 app.py 同層。")
        else:
            value_col = detect_shap_value_column(target_df)

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Model", display_model_name(shap_model), MODEL_NAMES[shap_model])
            with c2:
                metric_card("Target", target, TARGET_LABELS.get(target, ""))
            with c3:
                metric_card("Top Feature", target_df.sort_values(value_col, ascending=False).iloc[0]["feature"], "此 target 最重要特徵")

            col1, col2 = st.columns([1.25, 1])
            with col1:
                st.pyplot(
                    plot_barh(
                        target_df,
                        "feature",
                        value_col,
                        f"{display_model_name(shap_model)} SHAP Importance - {target}",
                    ),
                    use_container_width=True,
                )
            with col2:
                st.markdown("### Target Feature Ranking")
                st.dataframe(safe_format_df(target_df), use_container_width=True, hide_index=True)

            info_box(analyze_top_features(target_df, value_col, target=target, top_k=7), "blue")

            if "LSTM" in shap_model:
                info_box(
                    """
                    <b>LSTM SHAP 說明：</b><br>
                    LSTM 原始 SHAP 是針對 seq_len × features 的時間序列輸入計算，
                    本平台顯示的是將各 timestep 的同一 feature 聚合後的 feature-level importance。
                    """,
                    "yellow",
                )

            if target == "reward":
                info_box(
                    """
                    <b>Reward 特別解讀：</b><br>
                    若 reward 高度依賴 vehicle_type_ICE / EV / HEV，代表模型能區分不同車種在能耗、
                    排放或 reward function 中的差異。
                    """,
                    "purple",
                )

    elif shap_mode == "Model Comparison":
        compare_target = st.selectbox(
            "選擇比較目標",
            ["global"] + TARGETS,
            format_func=lambda x: "Global SHAP - All Targets" if x == "global" else f"{x}（{TARGET_LABELS.get(x, x)}）",
        )

        col1, col2 = st.columns(2)
        with col1:
            model_a = st.selectbox("模型 A", shap_models, index=0, format_func=lambda x: MODEL_NAMES[x])
        with col2:
            default_b = 1 if len(shap_models) > 1 else 0
            model_b = st.selectbox("模型 B", shap_models, index=default_b, format_func=lambda x: MODEL_NAMES[x])

        df_a, f_a = load_shap_csv(model_a, compare_target)
        df_b, f_b = load_shap_csv(model_b, compare_target)

        if df_a is None or df_b is None:
            st.error(f"找不到比較所需檔案：{f_a} 或 {f_b}")
        else:
            value_col = detect_shap_value_column(df_a)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### {display_model_name(model_a)}")
                st.dataframe(safe_format_df(df_a.head(10)), use_container_width=True, hide_index=True)
            with c2:
                st.markdown(f"### {display_model_name(model_b)}")
                st.dataframe(safe_format_df(df_b.head(10)), use_container_width=True, hide_index=True)

            fig, merged_df = plot_model_compare(
                df_a,
                df_b,
                display_model_name(model_a).replace(" ", "_"),
                display_model_name(model_b).replace(" ", "_"),
                value_col,
                f"{display_model_name(model_a)} vs {display_model_name(model_b)} SHAP Feature Importance",
                top_n=12,
            )
            st.pyplot(fig, use_container_width=True)

            st.markdown("### Difference Table")
            st.dataframe(safe_format_df(merged_df), use_container_width=True, hide_index=True)

            info_box(comparison_text(df_a, df_b, display_model_name(model_a), display_model_name(model_b), value_col), "purple")

    elif shap_mode == "Raw SHAP Summary":
        shap_model = st.selectbox("選擇模型", shap_models, format_func=lambda x: MODEL_NAMES[x])
        obj, filename = load_shap_pkl(shap_model)

        if obj is None:
            st.error(f"找不到 {filename}")
        else:
            if "shap_values" in obj:
                arr = np.array(obj["shap_values"])
                shape_text = str(arr.shape)
                format_text = obj.get("format", "MLP SHAP format")
            elif "shap_values_feature_level" in obj:
                arr = np.array(obj["shap_values_feature_level"])
                shape_text = str(arr.shape)
                format_text = obj.get("feature_level_format", "LSTM feature-level format")
            else:
                arr = None
                shape_text = "Unknown"
                format_text = "Unknown"

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("SHAP File", filename, "pkl raw values")
            with c2:
                metric_card("SHAP Shape", shape_text, format_text)
            with c3:
                if arr is not None and arr.ndim >= 2:
                    metric_card("Samples", arr.shape[1], "離線抽樣樣本數")
                else:
                    metric_card("Samples", "N/A", "")

            st.markdown("### Target Columns")
            st.write(obj.get("target_columns", []))

            st.markdown("### Input Columns")
            st.write(obj.get("input_columns", []))

            if "seq_len" in obj:
                st.markdown("### LSTM Sequence Info")
                st.write({"seq_len": obj.get("seq_len"), "aggregation": obj.get("aggregation")})

            info_box(
                """
                <b>方法說明：</b><br>
                本 SHAP 為 sample-based approximation。
                對 LSTM 而言，原始輸入為 seq_len × 17 的 flattened sequence，
                此 app 顯示的是聚合到 17 個 feature 後的 feature-level importance。
                """,
                "yellow",
            )

    elif shap_mode == "XAI Findings":
        st.markdown("### XAI 發現")

        info_box(
            """
            <b>Finding 1：模型主要依賴合理交通狀態特徵</b><br>
            SHAP 顯示 target_speed、speed_t、dist_tls_t、time_loss_t、tls_color_t 等特徵常排名靠前。
            這表示模型主要依據速度、號誌距離、號誌狀態與延遲等交通因素進行預測。
            """,
            "green",
        )

        info_box(
            """
            <b>Finding 2：不同 target 對應不同合理特徵</b><br>
            speed_next 主要依賴 target_speed 與 speed_t；
            dist_tls_next 主要依賴 dist_tls_t；
            tls_color_next 主要依賴 tls_color_t；
            waiting_time_next 主要依賴 is_stopped_t 與 waiting_time_t。
            這些依賴關係符合交通物理與駕駛邏輯。
            """,
            "blue",
        )

        info_box(
            """
            <b>Finding 3：Reward prediction 具有車種敏感性</b><br>
            reward 的 SHAP 中，vehicle_type_ICE、vehicle_type_EV、vehicle_type_HEV 可能具有較明顯影響。
            這表示模型能捕捉不同車種在能耗、排放或 reward function 中的差異。
            """,
            "purple",
        )

        info_box(
            """
            <b>Finding 4：Source policy 影響相對有限</b><br>
            若 source_policy_DQN、source_policy_DDQN、source_policy_DDPG 的重要性較低，
            可解讀為模型沒有主要記住資料來自哪個 policy，而是在學較通用的交通動態。
            """,
            "gray",
        )

        info_box(
            """
            <b>Finding 5：LSTM SHAP 提供 temporal model 的 feature-level 解釋</b><br>
            LSTM SHAP 先對 seq_len × features 的時間序列輸入進行解釋，
            再聚合成 feature-level importance。這能幫助比較 MLP 與 LSTM 在交通狀態依賴上的差異。
            """,
            "yellow",
        )


elif page == "7. Research Findings":
    st.markdown('<div class="section-title">Research Findings and Presentation Notes</div>', unsafe_allow_html=True)

    info_box(
        """
        <b>Finding 1：v2 temporal features 改善多數 one-step 連續預測</b><br>
        MLP (Temporal) 在 speed_next、acceleration_next、dist_tls_next、leader_speed_next、time_loss_next 與 reward 等多數 MAE 指標上較佳，
        代表 temporal enhanced data 有助於模型描述交通狀態轉移。
        """,
        "green",
    )

    info_box(
        """
        <b>Finding 2：LSTM 對特定交通狀態與長期 rollout 更穩定</b><br>
        LSTM (Temporal) 在 tls_color_next 與 waiting_time_next 表現最佳。
        在 multi-step rollout 中，LSTM (Temporal) 避免了 MLP (Temporal) 在 50-step 的明顯誤差發散。
        """,
        "blue",
    )

    info_box(
        """
        <b>Finding 3：SHAP 顯示模型依賴合理交通因素</b><br>
        真實 SHAP 結果顯示，模型對 speed_next、dist_tls_next、tls_color_next、waiting_time_next 等 target 的特徵依賴，
        大致符合交通物理與駕駛邏輯。
        """,
        "purple",
    )

    info_box(
        """
        <b>Finding 4：Reward 與車種特徵相關</b><br>
        reward 的 SHAP 分析中，vehicle_type 具有較高重要性，代表模型捕捉了 ICE、EV、HEV 在 reward / 能耗 / 排放機制上的差異。
        """,
        "yellow",
    )

    info_box(
        """
        <b>Future Work：Dreamer / Latent World Model</b><br>
        即使 LSTM 改善 long-horizon prediction stability，長期 rollout 仍可能存在 prediction drift。
        因此下一步可導入 Dreamer，在 latent space 中進行 imagination rollout，
        減少 observation-space 直接預測造成的 error accumulation。
        """,
        "red",
    )

    with st.expander("結論"):
        st.markdown(
            """
            本研究將智慧交通車速控制問題轉換為 world model prediction task，
            也就是根據目前交通狀態與控制動作預測下一步交通狀態。

            我比較了四個模型：MLP (Baseline)、LSTM (Baseline)、MLP (Temporal) 與 LSTM (Temporal)。
            Baseline 是 static baseline dataset，而 Temporal 加入 temporal context，
            讓資料保留 sequence 與 episode 資訊。

            在 one-step evaluation 中，MLP (Temporal) 在多數連續變數上有較低 MAE，
            顯示 temporal enhanced data 對單步預測有幫助。
            但 LSTM 在 tls_color、waiting_time 等狀態判斷上具有優勢。

            在 multi-step rollout 中，MLP (Temporal) 在長 horizon 下產生明顯 error accumulation，
            而 LSTM (Temporal) 能維持較穩定的長期預測。
            這說明 temporal memory 對 world model 的 long-horizon imagination 非常重要。

            XAI 分析使用離線計算的真實 SHAP values。
            結果顯示，模型主要依賴 target_speed、speed_t、dist_tls_t、tls_color_t、time_loss_t 等合理交通因素。
            對於 reward，模型也會依賴 vehicle_type，代表它捕捉不同車種在能耗或 reward mechanism 上的差異。

            最後，由於長期 rollout 仍可能存在 prediction drift，
            因此後續研究可導入 Dreamer 或 latent world model，
            在 latent space 中進行更穩定的 imagination rollout。
            """
        )
