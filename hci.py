# hci_world_model_app.py
# ============================================================
# HCI x Explainable World Model Platform
# 單頁式受試者測試流程版
#
# 執行：
#   streamlit run hci_world_model_app.py
#
# 功能：
# 1. Participant ID 在主畫面輸入，不在 sidebar
# 2. 沒有情境下拉選單，受試者只能依序答題
# 3. 每個情境流程：
#    無提示介面 → 交通判斷 3 題
#    有 AI 提示介面 → 相同交通判斷 3 題
#    比較問卷 → 進入下一個情境
# 4. 四個智慧交通情境
# 5. 自動記錄答案到 hci_world_model_logs.csv
# 6. Researcher Dashboard 可看統計與下載資料
# ============================================================

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="HCI x Explainable World Model Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOG_PATH = Path("hci_world_model_logs.csv")


# ============================================================
# Style
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #172033;
        font-weight: 900 !important;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 2.0rem 2.3rem;
        border-radius: 0 0 24px 24px;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 900;
        margin-bottom: 0.45rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.95;
        line-height: 1.75;
    }
    .card {
        background: white;
        border-radius: 18px;
        padding: 1.3rem 1.45rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .blue-card {
        background: #eff6ff;
        border-left: 6px solid #2563eb;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        line-height: 1.8;
    }
    .yellow-card {
        background: #fffbeb;
        border-left: 6px solid #d97706;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        line-height: 1.8;
    }
    .green-card {
        background: #ecfdf5;
        border-left: 6px solid #059669;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        line-height: 1.8;
    }
    .red-card {
        background: #fef2f2;
        border-left: 6px solid #dc2626;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        line-height: 1.8;
    }
    .purple-card {
        background: #f5f3ff;
        border-left: 6px solid #7c3aed;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        line-height: 1.8;
    }
    .section-title {
        font-size: 1.65rem;
        font-weight: 900;
        color: #111827;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }
    .task-title {
        font-size: 1.45rem;
        font-weight: 900;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    .condition-pill {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 850;
        color: white;
        background: #2563eb;
        margin-bottom: 0.8rem;
    }
    .condition-pill.noexp {
        background: #6b7280;
    }
    .condition-pill.exp {
        background: #7c3aed;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .metric-label {
        color: #6b7280;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 900;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Scenario Data
# ============================================================

SCENARIOS = [
    {
        "id": "S1",
        "title": "情境 1：紅燈接近，系統建議減速",
        "summary": "車輛距離號誌路口 38 m，目前號誌為紅燈，車速 42 km/h，前方無明顯阻擋。",
        "state": {
            "speed_t": "42 km/h",
            "acceleration_t": "0.3 m/s²",
            "dist_tls_t": "38 m",
            "tls_color_t": "Red",
            "leader_gap_t": "32 m",
            "leader_speed_t": "35 km/h",
            "waiting_time_t": "0 sec",
            "time_loss_t": "4.1 sec",
        },
        "recommendation": "建議減速至 25 km/h",
        "prediction": {
            "speed_next": "31.4 km/h",
            "dist_tls_next": "29.1 m",
            "waiting_time_next": "0.6 sec",
            "time_loss_next": "5.2 sec",
            "risk_level": "Medium",
        },
        "explanation": [
            ("dist_tls_t 距離號誌過近", "紅燈接近時若維持高速，下一步停等與 time loss 可能增加。", 42),
            ("tls_color_t 為紅燈", "號誌狀態使減速決策更合理。", 31),
            ("speed_t 目前速度偏高", "目前車速相對接近路口距離偏高，因此模型傾向建議減速。", 18),
        ],
    },
    {
        "id": "S2",
        "title": "情境 2：前車距離過短，系統建議減速",
        "summary": "車輛距離路口 92 m，號誌為綠燈，但前車距離只有 8 m，前車速度較慢。",
        "state": {
            "speed_t": "38 km/h",
            "acceleration_t": "0.1 m/s²",
            "dist_tls_t": "92 m",
            "tls_color_t": "Green",
            "leader_gap_t": "8 m",
            "leader_speed_t": "18 km/h",
            "waiting_time_t": "0 sec",
            "time_loss_t": "2.4 sec",
        },
        "recommendation": "建議減速至 22 km/h",
        "prediction": {
            "speed_next": "27.2 km/h",
            "dist_tls_next": "84.8 m",
            "waiting_time_next": "0.0 sec",
            "time_loss_next": "2.8 sec",
            "risk_level": "Medium",
        },
        "explanation": [
            ("leader_gap_t 前車距離過短", "前車距離是主要安全因素，車距不足時減速較合理。", 46),
            ("leader_speed_t 前車速度較慢", "前車速度偏低，若不減速可能造成跟車風險。", 29),
            ("speed_t 目前速度較高", "目前速度與前車距離不匹配，因此模型降低目標速度。", 17),
        ],
    },
    {
        "id": "S3",
        "title": "情境 3：綠燈且距離充足，系統建議小幅加速",
        "summary": "車輛距離路口 130 m，號誌為綠燈，前方車距充足，系統預測可順利通過。",
        "state": {
            "speed_t": "31 km/h",
            "acceleration_t": "0.0 m/s²",
            "dist_tls_t": "130 m",
            "tls_color_t": "Green",
            "leader_gap_t": "55 m",
            "leader_speed_t": "40 km/h",
            "waiting_time_t": "0 sec",
            "time_loss_t": "1.5 sec",
        },
        "recommendation": "建議小幅加速至 38 km/h",
        "prediction": {
            "speed_next": "35.6 km/h",
            "dist_tls_next": "120.4 m",
            "waiting_time_next": "0.0 sec",
            "time_loss_next": "1.2 sec",
            "risk_level": "Low",
        },
        "explanation": [
            ("tls_color_t 為綠燈", "目前號誌允許通行，系統傾向維持效率。", 37),
            ("dist_tls_t 距離充足", "距離路口仍有足夠反應空間，小幅加速風險較低。", 28),
            ("leader_gap_t 前車距離充足", "前方車距較安全，因此採用較積極的速度建議。", 22),
        ],
    },
    {
        "id": "S4",
        "title": "情境 4：長期預測不穩定，使用者應謹慎判斷",
        "summary": "車輛接近複雜路口，號誌即將變化，前方車流不穩定，World Model 的長期 rollout 不確定性較高。",
        "state": {
            "speed_t": "45 km/h",
            "acceleration_t": "0.4 m/s²",
            "dist_tls_t": "68 m",
            "tls_color_t": "Yellow",
            "leader_gap_t": "14 m",
            "leader_speed_t": "28 km/h",
            "waiting_time_t": "0 sec",
            "time_loss_t": "6.8 sec",
        },
        "recommendation": "建議減速至 28 km/h，並保持觀察",
        "prediction": {
            "speed_next": "34.5 km/h",
            "dist_tls_next": "59.3 m",
            "waiting_time_next": "0.3 sec",
            "time_loss_next": "8.6 sec",
            "risk_level": "High uncertainty",
        },
        "explanation": [
            ("tls_color_t 號誌為黃燈", "號誌即將轉換，使下一步狀態較不確定。", 35),
            ("leader_gap_t 前車距離偏短", "前方車流變化可能影響安全與速度控制。", 27),
            ("time_loss_t 已偏高", "目前 time loss 累積偏高，長期預測可能出現誤差累積。", 25),
        ],
    },
]

PHASES = ["no_explanation", "with_explanation", "comparison"]

TRAFFIC_QUESTIONS = [
    {
        "key": "reasonableness",
        "label": "Q1. 你認為 World Model 的建議在此情境下是合理的嗎？",
        "options": ["非常不合理", "不太合理", "普通", "合理", "非常合理"],
    },
    {
        "key": "accept_intention",
        "label": "Q2. 如果你是駕駛，你會採用這個 AI 建議嗎？",
        "options": ["完全不會", "不太會", "不確定", "會", "非常會"],
    },
    {
        "key": "decision",
        "label": "Q3. 你的最終操作選擇是什麼？",
        "options": ["Accept：採用 AI 建議", "Override：我會改用自己的判斷"],
    },
]

COMPARISON_QUESTIONS = [
    {
        "key": "understanding_compare",
        "label": "Q1. 哪個介面讓你比較理解 AI 為什麼做出這個建議？",
        "options": ["無提示介面", "有 AI 提示介面", "兩者差不多"],
    },
    {
        "key": "trust_compare",
        "label": "Q2. 哪個介面讓你比較信任 AI 的建議？",
        "options": ["無提示介面", "有 AI 提示介面", "兩者差不多"],
    },
    {
        "key": "helpfulness",
        "label": "Q3. AI 提示是否幫助你做出交通判斷？",
        "options": ["完全沒有幫助", "幫助很少", "普通", "有幫助", "非常有幫助"],
    },
    {
        "key": "cognitive_load",
        "label": "Q4. AI 提示是否讓你覺得資訊太多或造成負擔？",
        "options": ["完全不會", "不太會", "普通", "會", "非常會"],
    },
]


# ============================================================
# Session State
# ============================================================

def init_state():
    defaults = {
        "started": False,
        "finished": False,
        "participant_id": "",
        "scenario_idx": 0,
        "phase_idx": 0,
        "step_start_time": None,
        "local_records": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_experiment():
    st.session_state.started = False
    st.session_state.finished = False
    st.session_state.participant_id = ""
    st.session_state.scenario_idx = 0
    st.session_state.phase_idx = 0
    st.session_state.step_start_time = None
    st.session_state.local_records = []


def start_experiment(pid: str):
    st.session_state.started = True
    st.session_state.finished = False
    st.session_state.participant_id = pid.strip()
    st.session_state.scenario_idx = 0
    st.session_state.phase_idx = 0
    st.session_state.step_start_time = time.time()
    st.session_state.local_records = []


def get_current_step():
    scenario = SCENARIOS[st.session_state.scenario_idx]
    phase = PHASES[st.session_state.phase_idx]
    return scenario, phase


def total_steps():
    return len(SCENARIOS) * len(PHASES)


def current_step_number():
    return st.session_state.scenario_idx * len(PHASES) + st.session_state.phase_idx + 1


def phase_label(phase):
    if phase == "no_explanation":
        return "無提示介面"
    if phase == "with_explanation":
        return "有 AI 提示介面"
    return "比較問卷"


# ============================================================
# Logging
# ============================================================

def append_log(record: dict):
    df_new = pd.DataFrame([record])
    if LOG_PATH.exists():
        df_old = pd.read_csv(LOG_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def load_logs():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


# ============================================================
# Rendering Helpers
# ============================================================

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🧠 HCI × Explainable World Model Platform</div>
            <div class="hero-subtitle">
                單頁式 HCI 受試者測試平台：依序比較「無提示」與「有 AI 提示」對智慧交通判斷、信任與採用意願的影響。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress():
    step = current_step_number()
    total = total_steps()
    st.progress(step / total)
    st.caption(f"目前進度：Step {step} / {total}　｜　Participant ID：{st.session_state.participant_id}")


def render_state_table(scenario):
    state_df = pd.DataFrame(
        [{"Feature": k, "Value": v} for k, v in scenario["state"].items()]
    )
    st.dataframe(state_df, use_container_width=True, hide_index=True)


def render_prediction_table(scenario):
    pred_df = pd.DataFrame(
        [{"Prediction Target": k, "Predicted Value": v} for k, v in scenario["prediction"].items()]
    )
    st.dataframe(pred_df, use_container_width=True, hide_index=True)


def render_explanation(scenario):
    st.markdown("### 🤖 AI 提示 / Explanation")
    st.markdown(
        """
        <div class="purple-card">
        此提示顯示 World Model 判斷時較重要的交通因素。百分比為示意化的重要程度，用於協助受試者理解 AI 建議原因。
        </div>
        """,
        unsafe_allow_html=True,
    )
    for feature, desc, pct in scenario["explanation"]:
        st.markdown(f"**{feature}**：{desc}")
        st.progress(pct / 100)
        st.caption(f"重要程度：約 {pct}%")


def render_scenario_interface(scenario, phase):
    no_exp = phase == "no_explanation"
    pill_class = "noexp" if no_exp else "exp"
    pill_text = "無提示介面：只顯示交通狀態與 AI 建議" if no_exp else "有 AI 提示介面：顯示交通狀態、AI 建議與原因提示"

    st.markdown(f'<span class="condition-pill {pill_class}">{pill_text}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="task-title">{scenario["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="blue-card"><b>情境描述：</b>{scenario["summary"]}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 1])
    with col1:
        st.markdown("### 🚦 Traffic State")
        render_state_table(scenario)
    with col2:
        st.markdown("### 🎯 World Model Recommendation")
        st.markdown(
            f"""
            <div class="yellow-card">
            <b>World Model 建議：</b>{scenario['recommendation']}<br><br>
            <b>預測下一步狀態：</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_prediction_table(scenario)

    if phase == "with_explanation":
        render_explanation(scenario)
    else:
        st.markdown(
            """
            <div class="red-card">
            本階段不提供 AI 原因提示。請你只根據交通狀態與 World Model 建議做判斷。
            </div>
            """,
            unsafe_allow_html=True,
        )


def next_step():
    if st.session_state.phase_idx < len(PHASES) - 1:
        st.session_state.phase_idx += 1
    else:
        st.session_state.phase_idx = 0
        st.session_state.scenario_idx += 1

    if st.session_state.scenario_idx >= len(SCENARIOS):
        st.session_state.finished = True
        st.session_state.started = False
    else:
        st.session_state.step_start_time = time.time()


# ============================================================
# Pages
# ============================================================

def render_landing_page():
    st.markdown('<div class="section-title">受試者測試開始</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="green-card">
        請先輸入 Participant ID，按下「開始測試」後，系統會自動依序顯示四個智慧交通情境。<br>
        每個情境會先看到「無提示介面」，再看到「有 AI 提示介面」，最後回答該情境的比較問卷。<br>
        測試過程中不需要也不能自行選擇情境，請依照畫面順序完成。
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        pid = st.text_input("Participant ID", placeholder="例如 P001 或學號", key="pid_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        start_clicked = st.button("開始測試", type="primary", use_container_width=True)

    if start_clicked:
        if not pid.strip():
            st.error("請先輸入 Participant ID，才可以開始測試。")
        else:
            start_experiment(pid)
            st.rerun()

    st.markdown("### 測試流程")
    flow_df = pd.DataFrame(
        [
            ["1", "情境 1 無提示", "回答 3 題交通判斷"],
            ["2", "情境 1 有 AI 提示", "回答相同 3 題交通判斷"],
            ["3", "情境 1 比較問卷", "比較有無提示差異"],
            ["4", "情境 2～4", "重複相同流程"],
            ["5", "完成測試", "系統儲存作答資料"],
        ],
        columns=["Step", "階段", "內容"],
    )
    st.dataframe(flow_df, use_container_width=True, hide_index=True)


def render_traffic_questions(scenario, phase):
    form_key = f"form_{scenario['id']}_{phase}_{current_step_number()}"
    with st.form(form_key):
        st.markdown("### 請回答以下 3 題交通判斷題")
        answers = {}
        for q in TRAFFIC_QUESTIONS:
            answers[q["key"]] = st.radio(
                q["label"],
                q["options"],
                index=None,
                key=f"{form_key}_{q['key']}",
            )

        override_speed = ""
        if answers.get("decision") == "Override：我會改用自己的判斷":
            override_speed = st.text_input("如果你選擇 Override，你會建議改成多少速度或如何操作？", placeholder="例如：降到 18 km/h / 維持 30 km/h / 先觀察")

        submitted = st.form_submit_button("提交並進入下一步", type="primary", use_container_width=True)

    if submitted:
        missing = [q["label"] for q in TRAFFIC_QUESTIONS if answers.get(q["key"]) is None]
        if missing:
            st.error("請完成所有題目後再送出。")
            return

        response_time = round(time.time() - st.session_state.step_start_time, 3) if st.session_state.step_start_time else None
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "participant_id": st.session_state.participant_id,
            "scenario_id": scenario["id"],
            "scenario_title": scenario["title"],
            "phase": phase,
            "phase_label": phase_label(phase),
            "recommendation": scenario["recommendation"],
            "reasonableness": answers["reasonableness"],
            "accept_intention": answers["accept_intention"],
            "decision": answers["decision"],
            "override_speed_or_action": override_speed,
            "understanding_compare": "",
            "trust_compare": "",
            "helpfulness": "",
            "cognitive_load": "",
            "response_time_sec": response_time,
        }
        append_log(record)
        st.session_state.local_records.append(record)
        next_step()
        st.rerun()


def render_comparison_questions(scenario):
    st.markdown(f'<div class="task-title">{scenario["title"]}：有無提示比較問卷</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="green-card">
        你剛剛已經看過同一個交通情境的「無提示介面」與「有 AI 提示介面」。
        請根據剛才的體驗回答以下問題。
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_key = f"comparison_{scenario['id']}_{current_step_number()}"
    with st.form(form_key):
        answers = {}
        for q in COMPARISON_QUESTIONS:
            answers[q["key"]] = st.radio(
                q["label"],
                q["options"],
                index=None,
                key=f"{form_key}_{q['key']}",
            )
        comment = st.text_area("補充說明：你覺得 AI 提示哪裡有幫助或哪裡容易造成混淆？", placeholder="可不填")
        submitted = st.form_submit_button("提交並進入下一個情境", type="primary", use_container_width=True)

    if submitted:
        missing = [q["label"] for q in COMPARISON_QUESTIONS if answers.get(q["key"]) is None]
        if missing:
            st.error("請完成所有題目後再送出。")
            return

        response_time = round(time.time() - st.session_state.step_start_time, 3) if st.session_state.step_start_time else None
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "participant_id": st.session_state.participant_id,
            "scenario_id": scenario["id"],
            "scenario_title": scenario["title"],
            "phase": "comparison",
            "phase_label": "比較問卷",
            "recommendation": scenario["recommendation"],
            "reasonableness": "",
            "accept_intention": "",
            "decision": "",
            "override_speed_or_action": "",
            "understanding_compare": answers["understanding_compare"],
            "trust_compare": answers["trust_compare"],
            "helpfulness": answers["helpfulness"],
            "cognitive_load": answers["cognitive_load"],
            "comment": comment,
            "response_time_sec": response_time,
        }
        append_log(record)
        st.session_state.local_records.append(record)
        next_step()
        st.rerun()


def render_task_page():
    render_progress()
    scenario, phase = get_current_step()

    st.markdown('<div class="section-title">受試者任務</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="blue-card">
        目前階段：<b>{scenario['id']}｜{phase_label(phase)}</b><br>
        請依序完成本頁題目。系統會自動帶你進入下一步，不需要自行選擇情境。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if phase in ["no_explanation", "with_explanation"]:
        render_scenario_interface(scenario, phase)
        render_traffic_questions(scenario, phase)
    else:
        render_comparison_questions(scenario)


def render_finished_page():
    st.markdown('<div class="section-title">測試完成</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="green-card">
        感謝你完成本次 HCI × Explainable World Model 使用者測試。你的作答資料已儲存。
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.local_records:
        df = pd.DataFrame(st.session_state.local_records)
        st.markdown("### 本次作答摘要")
        st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("重新開始新的受試者測試", type="primary"):
        reset_experiment()
        st.rerun()


def render_researcher_dashboard():
    st.markdown('<div class="section-title">Researcher Dashboard</div>', unsafe_allow_html=True)
    df = load_logs()
    if df.empty:
        st.info("目前尚無作答資料。")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("總紀錄數", len(df))
    with c2:
        st.metric("受試者數", df["participant_id"].nunique() if "participant_id" in df else 0)
    with c3:
        st.metric("情境數", df["scenario_id"].nunique() if "scenario_id" in df else 0)
    with c4:
        traffic_df = df[df["phase"].isin(["no_explanation", "with_explanation"])]
        if not traffic_df.empty:
            accept_rate = (traffic_df["decision"] == "Accept：採用 AI 建議").mean() * 100
            st.metric("整體 Accept Rate", f"{accept_rate:.1f}%")
        else:
            st.metric("整體 Accept Rate", "N/A")

    st.markdown("### 原始資料")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下載 CSV",
        data=csv,
        file_name="hci_world_model_logs.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### 有無提示比較")
    traffic_df = df[df["phase"].isin(["no_explanation", "with_explanation"])].copy()
    if not traffic_df.empty:
        summary = traffic_df.groupby(["phase_label", "scenario_id"]).agg(
            records=("participant_id", "count"),
            avg_response_time=("response_time_sec", "mean"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

        accept_summary = traffic_df.copy()
        accept_summary["accept"] = accept_summary["decision"].eq("Accept：採用 AI 建議")
        accept_rate = accept_summary.groupby(["phase_label", "scenario_id"])["accept"].mean().reset_index()
        accept_rate["accept_rate_percent"] = accept_rate["accept"] * 100
        st.dataframe(accept_rate[["phase_label", "scenario_id", "accept_rate_percent"]], use_container_width=True, hide_index=True)

    comparison_df = df[df["phase"] == "comparison"].copy()
    if not comparison_df.empty:
        st.markdown("### 比較問卷結果")
        for col in ["understanding_compare", "trust_compare", "helpfulness", "cognitive_load"]:
            if col in comparison_df.columns:
                st.markdown(f"#### {col}")
                st.dataframe(comparison_df[col].value_counts().reset_index().rename(columns={"index": "answer", col: "count"}), use_container_width=True, hide_index=True)


def render_report_text():
    st.markdown('<div class="section-title">可放進報告的研究設計文字</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### HCI 實驗設計說明

        本研究以智慧交通 World Model 為核心，設計一個單頁式 HCI 使用者測試平台，
        目的在於評估「AI 解釋提示」是否能影響使用者對智慧交通決策建議的理解、信任與採用意願。

        實驗採用 within-subject design。每位受試者依序完成四個智慧交通情境，
        每個情境包含兩種介面條件：無提示介面與有 AI 提示介面。
        無提示介面僅顯示交通狀態與 World Model 建議；有 AI 提示介面則額外呈現模型判斷依據，
        例如號誌距離、號誌顏色、前車距離與目前車速等重要因素。

        在每個介面條件下，受試者需回答三個交通判斷題：
        第一，AI 建議是否合理；第二，是否願意採用 AI 建議；第三，最終會接受 AI 建議或選擇 override。
        完成同一情境的兩種介面後，受試者再回答比較問卷，用以衡量 AI 提示是否提升理解、信任與決策幫助，
        以及是否造成資訊負擔。

        本研究記錄的 HCI 指標包含：reasonableness、accept intention、accept / override decision、
        response time、perceived understanding、perceived trust、explanation helpfulness 與 cognitive load。
        透過比較有無 AI 提示條件下的回答差異，可分析 explanation 是否真的改善使用者對 World Model 的理解與採用行為。
        """
    )


# ============================================================
# Main App
# ============================================================

init_state()
render_hero()

# sidebar 只保留研究者功能，不放 participant id / condition / 情境選單
with st.sidebar:
    st.markdown("## 🧠 HCI Platform")
    mode = st.radio(
        "頁面",
        ["受試者測試", "Researcher Dashboard", "Report Text"],
        index=0,
    )
    st.divider()
    st.caption("受試者測試頁不提供情境選單，系統會自動依序進行。")

if mode == "受試者測試":
    if st.session_state.finished:
        render_finished_page()
    elif st.session_state.started:
        render_task_page()
    else:
        render_landing_page()
elif mode == "Researcher Dashboard":
    render_researcher_dashboard()
else:
    render_report_text()
