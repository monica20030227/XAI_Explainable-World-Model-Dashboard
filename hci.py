# hci_world_model_app_v2.py
# ============================================================
# HCI Experiment Platform for Explainable World Model
# 單頁式受試者測試流程版：
# 1. Participant Task 頁面內輸入 Participant ID
# 2. 按「開始測試」後，系統自動依序顯示 8 題
# 3. 受試者每題完成 Accept / Override 與量表後，自動進入下一題
# 4. 完成後顯示 Thank You Page
# 5. 研究者可到 Dashboard 查看統計並下載 CSV
#
# 執行：
#   streamlit run hci_world_model_app_v2.py
# ============================================================

import hashlib
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="HCI × Explainable World Model",
    page_icon="🧠",
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
        padding-top: 1.3rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        color: #111827;
        font-weight: 850 !important;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 2.0rem 2.2rem;
        border-radius: 24px;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.20);
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 900;
        margin-bottom: 0.45rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.96;
        line-height: 1.75;
    }
    .card {
        background: white;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .box {
        border-radius: 14px;
        padding: 1rem 1.15rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    .blue { background: #eff6ff; border-left: 6px solid #2563eb; }
    .green { background: #ecfdf5; border-left: 6px solid #059669; }
    .yellow { background: #fffbeb; border-left: 6px solid #d97706; }
    .red { background: #fef2f2; border-left: 6px solid #dc2626; }
    .purple { background: #f5f3ff; border-left: 6px solid #7c3aed; }
    .gray { background: #f9fafb; border-left: 6px solid #6b7280; }
    .big-number {
        font-size: 1.8rem;
        font-weight: 900;
        color: #111827;
    }
    .small-label {
        color: #6b7280;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .tag {
        display: inline-block;
        padding: 0.24rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 750;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        background: #e0f2fe;
        color: #075985;
    }
    .progress-text {
        font-size: 1.1rem;
        font-weight: 800;
        color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants
# ============================================================

LOG_PATH = Path("hci_world_model_logs.csv")

CONDITIONS = {
    "Without Explanation": {
        "name_zh": "無解釋介面",
        "description": "只顯示 World Model 的預測結果與建議，不提供 SHAP 或文字原因。",
    },
    "With Explanation": {
        "name_zh": "有解釋介面",
        "description": "除了預測結果，也顯示關鍵特徵、SHAP 重要性與自然語言解釋。",
    },
}

SCENARIOS = [
    {
        "scenario_id": "S1_red_light_close",
        "title": "情境 1：紅燈接近，系統建議減速",
        "traffic_context": "車輛距離號誌路口 38 m，目前號誌為紅燈，車速 42 km/h，前方無明顯阻擋。",
        "state": {
            "speed_t": 42,
            "acceleration_t": 0.3,
            "dist_tls_t": 38,
            "tls_color_t": "Red",
            "leader_gap_t": 32,
            "leader_speed_t": 35,
            "waiting_time_t": 0,
            "time_loss_t": 4.6,
            "vehicle_type": "ICE",
        },
        "ai_recommendation": "建議減速至 25 km/h",
        "prediction": {
            "speed_next": 31.4,
            "dist_tls_next": 29.1,
            "waiting_time_next": 0.6,
            "time_loss_next": 5.2,
            "risk_level": "Medium",
        },
        "explanation": [
            ("dist_tls_t", 0.31, "距離號誌太近，若維持高速可能造成急煞或停等不穩定。"),
            ("tls_color_t", 0.27, "目前是紅燈，模型預測需要提前降低速度。"),
            ("speed_t", 0.22, "目前速度偏高，因此影響下一步速度與 time loss。"),
            ("target_speed", 0.13, "控制目標速度會直接影響下一步狀態預測。"),
        ],
        "correct_behavior": "Accept",
        "teaching_note": "合理行為是接受減速建議，因為紅燈距離近且目前速度偏高。",
    },
    {
        "scenario_id": "S2_leader_gap_short",
        "title": "情境 2：前車距離過短，系統建議減速",
        "traffic_context": "車輛距離路口 92 m，號誌為綠燈，但前車距離只有 8 m，前車速度較慢。",
        "state": {
            "speed_t": 38,
            "acceleration_t": 0.1,
            "dist_tls_t": 92,
            "tls_color_t": "Green",
            "leader_gap_t": 8,
            "leader_speed_t": 18,
            "waiting_time_t": 0,
            "time_loss_t": 2.1,
            "vehicle_type": "HEV",
        },
        "ai_recommendation": "建議減速至 22 km/h",
        "prediction": {
            "speed_next": 27.2,
            "dist_tls_next": 84.8,
            "waiting_time_next": 0.0,
            "time_loss_next": 2.8,
            "risk_level": "Medium",
        },
        "explanation": [
            ("leader_gap_t", 0.34, "前車距離過短，是模型判斷需要減速的主要因素。"),
            ("leader_speed_t", 0.26, "前車速度較慢，若不減速可能造成跟車風險。"),
            ("speed_t", 0.18, "目前車速高於前車速度，因此模型預測需要降低速度。"),
            ("dist_tls_t", 0.10, "距離路口仍有一段距離，號誌因素不是主要原因。"),
        ],
        "correct_behavior": "Accept",
        "teaching_note": "雖然是綠燈，但前車距離過短，因此合理行為仍是接受減速建議。",
    },
    {
        "scenario_id": "S3_green_light_far",
        "title": "情境 3：綠燈且距離充足，系統建議小幅加速",
        "traffic_context": "車輛距離路口 145 m，號誌為綠燈，前方車距安全，系統建議小幅加速以降低 time loss。",
        "state": {
            "speed_t": 28,
            "acceleration_t": 0.0,
            "dist_tls_t": 145,
            "tls_color_t": "Green",
            "leader_gap_t": 46,
            "leader_speed_t": 34,
            "waiting_time_t": 0,
            "time_loss_t": 3.4,
            "vehicle_type": "EV",
        },
        "ai_recommendation": "建議加速至 35 km/h",
        "prediction": {
            "speed_next": 33.1,
            "dist_tls_next": 136.4,
            "waiting_time_next": 0.0,
            "time_loss_next": 2.9,
            "risk_level": "Low",
        },
        "explanation": [
            ("tls_color_t", 0.25, "目前為綠燈，模型預測通過路口機會較高。"),
            ("dist_tls_t", 0.23, "距離路口充足，因此不需要急煞或停等。"),
            ("leader_gap_t", 0.20, "前方車距安全，支援小幅加速。"),
            ("time_loss_t", 0.17, "小幅加速可降低 time loss。"),
        ],
        "correct_behavior": "Accept",
        "teaching_note": "合理行為是接受建議，但仍需注意不能過度加速。",
    },
    {
        "scenario_id": "S4_prediction_drift",
        "title": "情境 4：長期預測不穩定，使用者應謹慎判斷",
        "traffic_context": "World Model 在 50-step rollout 出現較高誤差累積，但系統仍給出明確建議。",
        "state": {
            "speed_t": 36,
            "acceleration_t": -0.2,
            "dist_tls_t": 66,
            "tls_color_t": "Yellow",
            "leader_gap_t": 18,
            "leader_speed_t": 21,
            "waiting_time_t": 0,
            "time_loss_t": 6.8,
            "vehicle_type": "ICE",
        },
        "ai_recommendation": "建議維持 36 km/h",
        "prediction": {
            "speed_next": 35.2,
            "dist_tls_next": 55.0,
            "waiting_time_next": 1.4,
            "time_loss_next": 8.9,
            "risk_level": "High",
        },
        "explanation": [
            ("rollout_horizon", 0.36, "長期 rollout 誤差可能累積，預測可靠度下降。"),
            ("tls_color_t", 0.24, "黃燈代表狀態變化快速，模型判斷不確定性較高。"),
            ("leader_gap_t", 0.19, "前車距離偏短，維持速度可能有風險。"),
            ("time_loss_t", 0.12, "time loss 已偏高，但不能只為降低延遲而忽略安全。"),
        ],
        "correct_behavior": "Override",
        "teaching_note": "此情境中合理行為是 override 或要求更多資訊，因為長期預測不穩定且風險較高。",
    },
]

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
    columns=["Target", "Metric", "MLP (Baseline)", "LSTM (Baseline)", "MLP (Temporal)", "LSTM (Temporal)"],
)

ROLLOUT_SUMMARY = pd.DataFrame(
    [
        [10, 0.3270, 0.3293, 56.9968, 25.5172, 7.5647, 3.0765],
        [20, 0.5750, 0.6540, 123.9111, 56.6645, 14.7849, 5.3105],
        [50, 72.5759, 0.5935, 7615.2311, 102.0483, 3167.6657, 10.9053],
    ],
    columns=[
        "Horizon",
        "MLP (Temporal) speed MAE",
        "LSTM (Temporal) speed MAE",
        "MLP (Temporal) dist_tls MAE",
        "LSTM (Temporal) dist_tls MAE",
        "MLP (Temporal) time_loss MAE",
        "LSTM (Temporal) time_loss MAE",
    ],
)


# ============================================================
# Helpers
# ============================================================

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🧠 HCI × Explainable World Model Platform</div>
            <div class="hero-subtitle">
                單頁式受試者測試流程：輸入 Participant ID → 開始測試 → 情境1無提示 → 情境1有提示 → 比較問卷 → 依序完成四個交通情境。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text, color="blue"):
    st.markdown(f'<div class="box {color}">{text}</div>', unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="small-label">{label}</div>
            <div class="big-number">{value}</div>
            <div style="color:#6b7280; line-height:1.6;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_tags(tags):
    html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])
    st.markdown(html, unsafe_allow_html=True)


def append_log(row):
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def load_logs():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


def plot_feature_importance(explanation):
    features = [x[0] for x in explanation]
    values = [x[1] for x in explanation]
    y = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Relative importance")
    ax.set_title("Explanation: Key Features", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_rollout():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        ROLLOUT_SUMMARY["Horizon"],
        ROLLOUT_SUMMARY["MLP (Temporal) speed MAE"],
        marker="o",
        linewidth=2,
        label="MLP (Temporal)",
    )
    ax.plot(
        ROLLOUT_SUMMARY["Horizon"],
        ROLLOUT_SUMMARY["LSTM (Temporal) speed MAE"],
        marker="o",
        linewidth=2,
        label="LSTM (Temporal)",
    )
    ax.set_xlabel("Rollout Horizon")
    ax.set_ylabel("speed_next MAE")
    ax.set_title("World Model Long-horizon Stability", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig


def _trial_logs(logs):
    if logs.empty:
        return logs
    if "stage_type" in logs.columns:
        return logs[logs["stage_type"].fillna("interface_trial") == "interface_trial"].copy()
    return logs.copy()


def compute_dashboard_metrics(logs):
    logs = _trial_logs(logs)
    if logs.empty:
        return {
            "n": 0,
            "adoption_rate": 0,
            "override_rate": 0,
            "avg_trust": 0,
            "avg_understanding": 0,
            "avg_time": 0,
            "correct_rate": 0,
        }

    return {
        "n": len(logs),
        "adoption_rate": (logs["decision"] == "Accept").mean() * 100 if "decision" in logs else 0,
        "override_rate": (logs["decision"] == "Override").mean() * 100 if "decision" in logs else 0,
        "avg_trust": pd.to_numeric(logs.get("trust_score", pd.Series(dtype=float)), errors="coerce").mean(),
        "avg_understanding": pd.to_numeric(logs.get("understanding_score", pd.Series(dtype=float)), errors="coerce").mean(),
        "avg_time": pd.to_numeric(logs.get("decision_time_sec", pd.Series(dtype=float)), errors="coerce").mean(),
        "correct_rate": pd.to_numeric(logs.get("is_correct_behavior", pd.Series(dtype=float)), errors="coerce").mean() * 100,
    }


def condition_comparison(logs):
    """
    Dashboard 用的保險版條件比較。
    可以相容舊版 CSV、空 CSV、缺欄位 CSV，避免 pandas groupby agg KeyError。
    """
    logs = _trial_logs(logs)
    if logs is None or logs.empty:
        return pd.DataFrame()

    logs = logs.copy()

    # 舊版紀錄可能只有 condition，沒有 condition_zh
    if "condition_zh" not in logs.columns:
        if "condition" in logs.columns:
            logs["condition_zh"] = logs["condition"].map({
                "Without Explanation": "無 AI 提示",
                "With Explanation": "有 AI 提示",
                "no_hint": "無 AI 提示",
                "with_hint": "有 AI 提示",
            }).fillna(logs["condition"].astype(str))
        else:
            logs["condition_zh"] = "未標記條件"

    # 所有 Dashboard 需要的欄位都先補齊
    required_defaults = {
        "reasonableness_score": np.nan,
        "adoption_intention_score": np.nan,
        "trust_score": np.nan,
        "understanding_score": np.nan,
        "confidence_score": np.nan,
        "decision": "",
        "decision_time_sec": np.nan,
        "is_correct_behavior": np.nan,
    }
    for col, default in required_defaults.items():
        if col not in logs.columns:
            logs[col] = default

    numeric_cols = [
        "reasonableness_score",
        "adoption_intention_score",
        "trust_score",
        "understanding_score",
        "confidence_score",
        "decision_time_sec",
        "is_correct_behavior",
    ]
    for col in numeric_cols:
        logs[col] = pd.to_numeric(logs[col], errors="coerce")

    logs["decision"] = logs["decision"].fillna("").astype(str)
    logs["condition_zh"] = logs["condition_zh"].fillna("未標記條件").astype(str)

    rows = []
    for condition_name, g in logs.groupby("condition_zh", dropna=False):
        rows.append({
            "condition_zh": condition_name,
            "responses": len(g),
            "reasonableness_mean": g["reasonableness_score"].mean(),
            "adoption_intention_mean": g["adoption_intention_score"].mean(),
            "trust_mean": g["trust_score"].mean(),
            "understanding_mean": g["understanding_score"].mean(),
            "confidence_mean": g["confidence_score"].mean(),
            "adoption_rate": (g["decision"] == "Accept").mean() * 100,
            "override_rate": (g["decision"] == "Override").mean() * 100,
            "decision_time_mean": g["decision_time_sec"].mean(),
            "correct_behavior_rate": g["is_correct_behavior"].mean() * 100,
        })

    return pd.DataFrame(rows)


def pair_comparison_summary(logs):
    if logs.empty or "stage_type" not in logs.columns:
        return pd.DataFrame()
    comp = logs[logs["stage_type"] == "pair_questionnaire"].copy()
    if comp.empty:
        return pd.DataFrame()
    return (
        comp.groupby("scenario_title")
        .agg(
            responses=("scenario_title", "count"),
            explanation_help_mean=("explanation_help_score", "mean"),
            cognitive_load_mean=("pair_cognitive_load", "mean"),
            prefer_with_explanation_rate=("preferred_interface", lambda x: (x == "有 AI 提示介面").mean() * 100),
        )
        .reset_index()
    )

def make_task_order(participant_id):
    """
    產生固定 paired flow：
    情境1無提示 → 情境1有提示 → 情境1比較問卷 → 情境2...
    不讓受試者自行選擇情境或介面條件。
    """
    tasks = []
    for idx in range(len(SCENARIOS)):
        tasks.append({"scenario_index": idx, "condition": "Without Explanation", "stage_type": "interface_trial"})
        tasks.append({"scenario_index": idx, "condition": "With Explanation", "stage_type": "interface_trial"})
        tasks.append({"scenario_index": idx, "condition": "Pair Comparison", "stage_type": "pair_questionnaire"})
    return tasks


def init_session_state():
    defaults = {
        "participant_id_input": "",
        "experiment_started": False,
        "experiment_finished": False,
        "current_task_index": 0,
        "task_order": [],
        "task_start_time": time.time(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_experiment(participant_id):
    st.session_state.participant_id_input = participant_id.strip()
    st.session_state.task_order = make_task_order(participant_id.strip())
    st.session_state.current_task_index = 0
    st.session_state.experiment_started = True
    st.session_state.experiment_finished = False
    st.session_state.task_start_time = time.time()


def reset_experiment():
    st.session_state.experiment_started = False
    st.session_state.experiment_finished = False
    st.session_state.current_task_index = 0
    st.session_state.task_order = []
    st.session_state.task_start_time = time.time()


def next_task_or_finish():
    st.session_state.current_task_index += 1
    if st.session_state.current_task_index >= len(st.session_state.task_order):
        st.session_state.experiment_finished = True
        st.session_state.experiment_started = False
    st.session_state.task_start_time = time.time()


def get_elapsed_seconds():
    return round(time.time() - st.session_state.task_start_time, 2)


def render_condition_badge(condition):
    if condition == "Pair Comparison":
        info_box("<b>目前階段：</b>同一情境的介面比較問卷<br>請回想剛剛的無提示與有 AI 提示兩個畫面，回答哪一種呈現方式比較有幫助。", "purple")
        return
    zh = CONDITIONS[condition]["name_zh"]
    color = "green" if condition == "With Explanation" else "gray"
    info_box(f"<b>目前介面條件：</b>{zh}<br>{CONDITIONS[condition]['description']}", color)


def render_scenario_content(scenario, condition):
    col_state, col_ai = st.columns([1, 1])

    with col_state:
        st.markdown("### 🚦 Traffic State")
        info_box(f"<b>{scenario['title']}</b><br>{scenario['traffic_context']}", "blue")
        state_df = pd.DataFrame([{"Feature": k, "Value": v} for k, v in scenario["state"].items()])
        st.dataframe(state_df, use_container_width=True, hide_index=True)

    with col_ai:
        st.markdown("### 🤖 AI Recommendation")
        risk_color = "red" if scenario["prediction"]["risk_level"] == "High" else "yellow"
        info_box(
            f"""
            <b>World Model 建議：</b>{scenario['ai_recommendation']}<br><br>
            <b>預測下一步狀態：</b><br>
            - speed_next：{scenario['prediction']['speed_next']} km/h<br>
            - dist_tls_next：{scenario['prediction']['dist_tls_next']} m<br>
            - waiting_time_next：{scenario['prediction']['waiting_time_next']} sec<br>
            - time_loss_next：{scenario['prediction']['time_loss_next']} sec<br>
            - risk_level：{scenario['prediction']['risk_level']}
            """,
            risk_color,
        )

    if condition == "With Explanation":
        st.markdown("### 🔍 AI 提示 / Explanation Panel")
        col_chart, col_text = st.columns([1, 1])
        with col_chart:
            st.pyplot(plot_feature_importance(scenario["explanation"]), use_container_width=True)
        with col_text:
            for feature, value, reason in scenario["explanation"]:
                info_box(f"<b>{feature}</b>（importance = {value:.2f}）<br>{reason}", "green")
            info_box("<b>提醒：</b>提示是輔助判斷，不代表一定要完全相信 AI。請仍依交通情境做決策。", "purple")
    else:
        st.markdown("### 🔒 無 AI 提示")
        info_box("此階段不顯示 SHAP 或文字原因。請只根據交通狀態、AI 建議與預測結果做判斷。", "gray")


def render_interface_trial(task):
    scenario = SCENARIOS[task["scenario_index"]]
    condition = task["condition"]
    total_tasks = len(st.session_state.task_order)
    task_no = st.session_state.current_task_index + 1
    scenario_no = task["scenario_index"] + 1
    phase_name = "無提示" if condition == "Without Explanation" else "有 AI 提示"

    st.markdown(f'<div class="progress-text">情境 {scenario_no} / {len(SCENARIOS)}：{phase_name}</div>', unsafe_allow_html=True)
    st.progress(task_no / total_tasks)
    render_condition_badge(condition)
    render_scenario_content(scenario, condition)

    st.markdown("### 🧩 交通判斷題")
    st.caption("無提示與有提示階段會問同一組交通判斷題，方便比較 AI 提示是否改變判斷。")

    form_key = f"trial_form_{task_no}_{scenario['scenario_id']}_{condition}"
    with st.form(form_key):
        q1 = st.slider("1. 你認為 AI 在這個交通情境中的建議合理嗎？", 1, 5, 3)
        q2 = st.slider("2. 你會採用 AI 的這個建議嗎？", 1, 5, 3)
        decision = st.radio(
            "3. 如果你是駕駛，你最後會怎麼做？",
            ["Accept", "Override"],
            format_func=lambda x: "接受 AI 建議" if x == "Accept" else "Override / 不接受 AI 建議",
            horizontal=True,
        )
        override_speed = None
        if decision == "Override":
            override_speed = st.number_input(
                "若 Override，你會建議速度為多少 km/h？",
                min_value=0.0,
                max_value=80.0,
                value=30.0,
                step=1.0,
            )
        reason = st.text_area("請簡短說明你的判斷原因", placeholder="例如：紅燈距離近，所以我會接受減速建議。")
        submitted = st.form_submit_button("提交並進入下一步", type="primary", use_container_width=True)

    if submitted:
        elapsed = get_elapsed_seconds()
        is_correct = int(decision == scenario["correct_behavior"])
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "participant_id": st.session_state.participant_id_input,
            "task_no": task_no,
            "total_tasks": total_tasks,
            "stage_type": "interface_trial",
            "scenario_pair_no": scenario_no,
            "condition": condition,
            "condition_zh": CONDITIONS[condition]["name_zh"],
            "scenario_id": scenario["scenario_id"],
            "scenario_title": scenario["title"],
            "ai_recommendation": scenario["ai_recommendation"],
            "risk_level": scenario["prediction"]["risk_level"],
            "reasonableness_score": q1,
            "adoption_intention_score": q2,
            "decision": decision,
            "override_speed": override_speed,
            "correct_behavior": scenario["correct_behavior"],
            "is_correct_behavior": is_correct,
            # 保留舊 dashboard 欄位名稱，避免其他功能壞掉
            "trust_score": q1,
            "understanding_score": q1,
            "confidence_score": q2,
            "cognitive_load": np.nan,
            "decision_time_sec": elapsed,
            "reason": reason,
            "researcher_note": scenario["teaching_note"],
            "preferred_interface": "",
            "explanation_help_score": np.nan,
            "pair_cognitive_load": np.nan,
            "comparison_reason": "",
        }
        append_log(row)
        next_task_or_finish()
        st.rerun()


def render_pair_questionnaire(task):
    scenario = SCENARIOS[task["scenario_index"]]
    total_tasks = len(st.session_state.task_order)
    task_no = st.session_state.current_task_index + 1
    scenario_no = task["scenario_index"] + 1

    st.markdown(f'<div class="progress-text">情境 {scenario_no} / {len(SCENARIOS)}：介面比較問卷</div>', unsafe_allow_html=True)
    st.progress(task_no / total_tasks)
    render_condition_badge("Pair Comparison")

    info_box(
        f"""
        你剛剛已完成同一個交通情境的兩種版本：<br>
        1. <b>無提示介面</b>：只看到交通狀態、AI 建議與預測。<br>
        2. <b>有 AI 提示介面</b>：多看到關鍵特徵與文字原因。<br><br>
        請針對 <b>{scenario['title']}</b> 回答下列比較問卷。
        """,
        "blue",
    )

    form_key = f"pair_form_{task_no}_{scenario['scenario_id']}"
    with st.form(form_key):
        prefer_understanding = st.radio(
            "1. 哪個介面讓你比較理解 AI 為什麼做出這個建議？",
            ["無提示介面", "有 AI 提示介面", "兩者差不多"],
            horizontal=True,
        )
        prefer_trust = st.radio(
            "2. 哪個介面讓你比較信任 AI 的建議？",
            ["無提示介面", "有 AI 提示介面", "兩者差不多"],
            horizontal=True,
        )
        help_score = st.slider("3. AI 提示對你做交通判斷有幫助嗎？", 1, 5, 3)
        load_score = st.slider("4. AI 提示會不會讓資訊變得太多、增加判斷負擔？", 1, 5, 3)
        comparison_reason = st.text_area("5. 請簡短說明你對有無提示差異的感受", placeholder="例如：有提示時比較知道模型為什麼建議減速。")
        submitted = st.form_submit_button("提交並進入下一個情境", type="primary", use_container_width=True)

    if submitted:
        elapsed = get_elapsed_seconds()
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "participant_id": st.session_state.participant_id_input,
            "task_no": task_no,
            "total_tasks": total_tasks,
            "stage_type": "pair_questionnaire",
            "scenario_pair_no": scenario_no,
            "condition": "Pair Comparison",
            "condition_zh": "介面比較問卷",
            "scenario_id": scenario["scenario_id"],
            "scenario_title": scenario["title"],
            "ai_recommendation": scenario["ai_recommendation"],
            "risk_level": scenario["prediction"]["risk_level"],
            "reasonableness_score": np.nan,
            "adoption_intention_score": np.nan,
            "decision": "",
            "override_speed": np.nan,
            "correct_behavior": scenario["correct_behavior"],
            "is_correct_behavior": np.nan,
            "trust_score": np.nan,
            "understanding_score": np.nan,
            "confidence_score": np.nan,
            "cognitive_load": np.nan,
            "decision_time_sec": elapsed,
            "reason": "",
            "researcher_note": scenario["teaching_note"],
            "preferred_interface": prefer_understanding,
            "preferred_trust_interface": prefer_trust,
            "explanation_help_score": help_score,
            "pair_cognitive_load": load_score,
            "comparison_reason": comparison_reason,
        }
        append_log(row)
        next_task_or_finish()
        st.rerun()


def render_scenario_task(task):
    if task.get("stage_type") == "pair_questionnaire":
        render_pair_questionnaire(task)
    else:
        render_interface_trial(task)

def render_participant_start():
    st.markdown("## 受試者測試")
    info_box(
        """
        請先輸入 Participant ID，按下「開始測試」後，系統會在同一頁依序完成四個交通情境。<br>每個情境固定順序為：無提示 → 有 AI 提示 → 介面比較問卷。受試者不需要、也不能自行選擇情境或介面條件。
        """,
        "blue",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        pid = st.text_input(
            "Participant ID",
            value=st.session_state.participant_id_input,
            placeholder="例如 P001",
            key="pid_start_input",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        start = st.button("開始測試", type="primary", use_container_width=True)

    if start:
        if not pid.strip():
            st.error("請先輸入 Participant ID。")
        else:
            start_experiment(pid)
            st.rerun()

    st.markdown("### 測試內容")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Flow", "4 組", "每組：無提示 → 有提示 → 比較問卷")
    with c2:
        metric_card("Condition A", "無解釋", "只顯示預測與建議")
    with c3:
        metric_card("Condition B", "有解釋", "顯示關鍵特徵與文字解釋")
    with c4:
        metric_card("Output", "CSV", "自動儲存判斷與比較問卷")


def render_finished_page():
    st.markdown("## 測試完成")
    info_box(
        f"""
        感謝參與！Participant ID：<b>{st.session_state.participant_id_input}</b><br>
        你的完整測試作答已儲存到 <b>{LOG_PATH.name}</b>。
        """,
        "green",
    )

    logs = load_logs()
    if not logs.empty:
        user_logs = logs[logs["participant_id"] == st.session_state.participant_id_input]
        if not user_logs.empty:
            st.markdown("### 你的作答摘要")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Responses", len(user_logs), "完成題數")
            with c2:
                metric_card("Accept Rate", f"{(user_logs['decision'] == 'Accept').mean() * 100:.1f}%", "接受 AI 建議比例")
            with c3:
                metric_card("Avg Trust", f"{user_logs['trust_score'].mean():.2f}/5", "平均信任分數")
            with c4:
                metric_card("Avg Understanding", f"{user_logs['understanding_score'].mean():.2f}/5", "平均理解分數")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("重新開始另一位受試者", use_container_width=True):
            st.session_state.participant_id_input = ""
            reset_experiment()
            st.rerun()
    with col2:
        if st.button("回到開始畫面", use_container_width=True):
            reset_experiment()
            st.rerun()


# ============================================================
# Main App
# ============================================================

init_session_state()

with st.sidebar:
    st.markdown("## 🧠 HCI Platform")
    page = st.radio(
        "選擇頁面",
        [
            "1. Research Overview",
            "2. Study Design",
            "3. Participant Task",
            "4. Researcher Dashboard",
            "5. World Model Evidence",
            "6. Report Text",
        ],
        index=2,
    )
    st.divider()
    st.markdown("### 🧠 平台連結")

    st.image(
        "平台連結.png",
        use_container_width=True
    )

    st.caption("請掃描 QR Code 進入平台")

render_hero()

if page == "1. Research Overview":
    st.markdown("## 研究主題")
    info_box(
        """
        <b>題目：</b>以可解釋 World Model 介面提升智慧交通 AI 決策理解與信任之 HCI 研究。<br><br>
        本平台不是單純展示模型效能，而是讓使用者在交通情境中實際判斷是否採用 AI 建議，
        並比較「無解釋」與「有解釋」介面對 trust、understanding、adoption、override 的影響。
        """,
        "blue",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("AI Core", "World Model", "預測 sₜ + aₜ → sₜ₊₁")
    with c2:
        metric_card("HCI Manipulation", "With / Without XAI", "比較有無解釋介面")
    with c3:
        metric_card("Behavior Metrics", "Adoption / Override", "觀察是否接受 AI 建議")
    with c4:
        metric_card("Attitude Metrics", "Trust / Understanding", "Likert 量表")

    st.markdown("### 研究問題")
    rq_df = pd.DataFrame(
        [
            ["RQ1", "World Model explanation 是否提升使用者對 AI 建議的理解？"],
            ["RQ2", "World Model explanation 是否提升使用者信任？"],
            ["RQ3", "Explanation 是否降低不必要的 override，並改善 trust calibration？"],
            ["RQ4", "當模型長期 rollout 不穩定時，解釋是否能幫助使用者辨識風險？"],
        ],
        columns=["Research Question", "Description"],
    )
    st.dataframe(rq_df, use_container_width=True, hide_index=True)
    show_tags(["HCI Experiment", "World Model", "XAI", "Trust Calibration", "Override Behavior"])

elif page == "2. Study Design":
    st.markdown("## HCI 實驗設計")
    col1, col2 = st.columns(2)
    with col1:
        info_box(
            """
            <b>Condition A：Without Explanation</b><br>
            - 顯示交通狀態<br>
            - 顯示 AI 建議<br>
            - 顯示 World Model 預測結果<br>
            - 不顯示 SHAP 或原因說明
            """,
            "gray",
        )
    with col2:
        info_box(
            """
            <b>Condition B：With Explanation</b><br>
            - 顯示交通狀態<br>
            - 顯示 AI 建議<br>
            - 顯示 World Model 預測結果<br>
            - 顯示關鍵特徵、SHAP-like importance 與文字解釋
            """,
            "green",
        )

    st.markdown("### 單頁式測試流程")
    flow_df = pd.DataFrame(
        [
            ["Step 1", "受試者在 Participant Task 頁輸入 Participant ID"],
            ["Step 2", "按下開始測試"],
            ["Step 3", "系統依序顯示：情境1無提示 → 情境1有提示 → 情境1比較問卷 → 情境2..."],
            ["Step 4", "每個無/有提示畫面回答三題交通判斷；每個情境結束後填寫介面比較問卷"],
            ["Step 5", "提交後自動進入下一題"],
            ["Step 6", "全部完成後顯示 Thank You Page 並儲存 CSV"],
        ],
        columns=["Step", "Description"],
    )
    st.dataframe(flow_df, use_container_width=True, hide_index=True)

    st.markdown("### 主要變數")
    variable_df = pd.DataFrame(
        [
            ["Independent Variable", "Interface condition", "Without Explanation vs With Explanation"],
            ["Behavior DV", "Adoption", "是否接受 AI 建議"],
            ["Behavior DV", "Override", "是否拒絕 AI 建議"],
            ["Attitude DV", "Trust", "我信任這個 AI 建議"],
            ["Attitude DV", "Understanding", "我理解 AI 為什麼做出這個建議"],
            ["Process Metric", "Decision time", "做出決策花費秒數"],
            ["Risk Metric", "Correct behavior", "在高風險情境中是否做出合理採用或 override"],
        ],
        columns=["Type", "Variable", "Operational Definition"],
    )
    st.dataframe(variable_df, use_container_width=True, hide_index=True)

elif page == "3. Participant Task":
    if st.session_state.experiment_finished:
        render_finished_page()
    elif not st.session_state.experiment_started:
        render_participant_start()
    else:
        current_task = st.session_state.task_order[st.session_state.current_task_index]
        render_scenario_task(current_task)

elif page == "4. Researcher Dashboard":
    st.markdown("## 研究者 Dashboard")
    logs = load_logs()
    if logs.empty:
        st.warning("目前尚未有任何受試者資料。請先到 Participant Task 完成至少一筆紀錄。")
    else:
        metrics = compute_dashboard_metrics(logs)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Responses", int(metrics["n"]), "總作答筆數")
        with c2:
            metric_card("Adoption Rate", f"{metrics['adoption_rate']:.1f}%", "接受 AI 建議比例")
        with c3:
            metric_card("Avg Trust", f"{metrics['avg_trust']:.2f}/5", "平均信任分數")
        with c4:
            metric_card("Avg Understanding", f"{metrics['avg_understanding']:.2f}/5", "平均理解分數")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            metric_card("Override Rate", f"{metrics['override_rate']:.1f}%", "拒絕 AI 建議比例")
        with c6:
            metric_card("Decision Time", f"{metrics['avg_time']:.1f}s", "平均決策秒數")
        with c7:
            metric_card("Correct Behavior", f"{metrics['correct_rate']:.1f}%", "合理採用 / override")
        with c8:
            metric_card("Participants", logs["participant_id"].nunique(), "不重複受試者數")

        st.markdown("### Condition Comparison")
        comp = condition_comparison(logs)
        st.dataframe(comp, use_container_width=True, hide_index=True)

        st.markdown("### Raw Logs")
        st.dataframe(logs, use_container_width=True, hide_index=True)

        csv = logs.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下載 HCI 實驗資料 CSV",
            data=csv,
            file_name="hci_world_model_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif page == "5. World Model Evidence":
    st.markdown("## World Model Evidence")
    info_box(
        """
        這一頁用來支撐 HCI 平台不是假介面，而是延續原本 World Model 研究。
        可以在報告中說明：HCI 介面中的 AI 建議與解釋，來自 World Model 預測、rollout 穩定性與 SHAP 解釋概念。
        """,
        "blue",
    )
    st.markdown("### One-step Evaluation")
    st.dataframe(ONE_STEP_DATA, use_container_width=True, hide_index=True)
    st.markdown("### Multi-step Rollout Evidence")
    st.dataframe(ROLLOUT_SUMMARY, use_container_width=True, hide_index=True)
    info_box(
        """
        <b>報告重點：</b><br>
        - MLP (Temporal) 在 one-step prediction 上表現較佳。<br>
        - LSTM (Temporal) 在 long-horizon rollout 上較穩定。<br>
        - 因此 HCI 平台中特別設計「長期預測不穩定」情境，觀察 explanation 是否能幫助使用者避免過度信任。
        """,
        "green",
    )

elif page == "6. Report Text":
    st.markdown("## HCI 期末報告")
    st.markdown("### 題目")
    st.code("以可解釋 World Model 介面提升智慧交通 AI 決策理解與信任之 HCI 研究", language="text")

    st.markdown("### 內容介紹")
    st.text_area(
        "內容介紹",
        value=(
            "本研究延續智慧交通車速控制中的 World Model 主題，設計一套 HCI 實驗平台，"
            "探討可解釋介面是否能提升使用者對 AI 決策建議的理解、信任與採用判斷。"
            "World Model 的任務是根據目前交通狀態與控制動作預測下一步交通狀態，"
            "例如下一步速度、距離號誌、等待時間與 time loss。"
            "然而，在真實人機互動情境中，使用者不只需要知道 AI 建議什麼，"
            "也需要理解 AI 為什麼做出該建議，並在模型可能不穩定或高風險情境中保留 override 能力。"
        ),
        height=160,
    )

    st.markdown("### 研究方法")
    st.text_area(
        "研究方法",
        value=(
            "本研究採用 within-subject 的 HCI 實驗設計，讓每位受試者依序完成無解釋介面與有解釋介面的交通情境任務。"
            "受試者在每個任務中需要判斷是否接受 World Model 的速度建議，或選擇 override。"
            "系統會記錄作答條件、情境、決策結果、決策時間、信任分數、理解分數、決策信心與認知負荷。"
            "最後比較有無解釋介面在 trust、understanding、adoption rate、override rate 與 correct behavior 上的差異。"
        ),
        height=160,
    )

    st.markdown("### 預期結果")
    st.text_area(
        "預期結果",
        value=(
            "預期有解釋介面能提升使用者對 AI 建議的理解與信任，並在高風險或模型不穩定情境中幫助使用者做出較合理的 override。"
            "然而，解釋資訊也可能提高使用者的認知負荷與決策時間。因此，本研究的重點不是單純追求 adoption rate 提升，"
            "而是觀察 explanation 是否能促進 trust calibration，也就是讓使用者在合理情境中信任 AI，在高風險情境中保留懷疑與人工介入能力。"
        ),
        height=160,
    )
