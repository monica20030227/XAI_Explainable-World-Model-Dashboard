# run_lstm_shap_offline.py
# ============================================================
# Offline SHAP for LSTM World Models
# 支援：
#   LSTM v2: world_model_lstm_sequence.pt + world_model_lstm_sequence_scalers.pkl + world_model_all_sequence.csv
#   LSTM v1: world_model_lstm.pt + world_model_lstm_scalers.pkl + world_model_all_sequence.csv
#
# 重點：
#   LSTM input shape = (batch, seq_len, input_dim)
#   Kernel SHAP 不直接吃 3D，因此本程式會 flatten 成：
#       (batch, seq_len * input_dim)
#   再於 predict_fn 內 reshape 回：
#       (batch, seq_len, input_dim)
#
# 輸出：
#   shap_outputs/lstm_v2_shap_values.pkl
#   shap_outputs/lstm_v2_global_feature_importance.csv
#   shap_outputs/lstm_v2_<target>_feature_importance.csv
#
# 注意：
#   這裡輸出的 feature importance 已經把各 timestep 聚合成 feature-level importance。
#   例如 speed_t 代表整個 sequence 中 speed_t 的平均 / 加總影響。
#
# 安裝：
#   pip install shap torch pandas numpy scikit-learn matplotlib joblib
#
# 建議先小參數測試：
#   python run_lstm_shap_offline.py --model lstm_v2 --background 5 --samples 10 --nsamples 30
#
# 報告用較穩設定：
#   python run_lstm_shap_offline.py --model lstm_v2 --background 20 --samples 100 --nsamples 100
#
# 若要 v1/v2 都跑：
#   python run_lstm_shap_offline.py --model both --background 20 --samples 100 --nsamples 100
# ============================================================

import os
import argparse
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================
# 1. LSTM Model Definition
# 會優先從 state_dict 推斷 hidden_dim / num_layers
# ============================================================

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
# 2. Config
# ============================================================

MODEL_CONFIGS = {
    "lstm_v1": {
        "display_name": "LSTM v1 - Sequential World Model",
        "model_path": "world_model_lstm.pt",
        "scaler_path": "world_model_lstm_scalers.pkl",
        # LSTM v1 若沒有專屬 sequence csv，也先用 sequence dataset 取序列
        "data_path": "world_model_all_sequence.csv",
        "output_prefix": "lstm_v1",
    },
    "lstm_v2": {
        "display_name": "LSTM v2 - Enhanced Sequential World Model",
        "model_path": "world_model_lstm_sequence.pt",
        "scaler_path": "world_model_lstm_sequence_scalers.pkl",
        "data_path": "world_model_all_sequence.csv",
        "output_prefix": "lstm_v2",
    },
}


# ============================================================
# 3. Utilities
# ============================================================

def load_scalers(scaler_path: str):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 scaler 檔案：{scaler_path}")

    scalers = joblib.load(scaler_path)

    required_keys = ["x_scaler", "y_scaler", "input_columns", "target_columns"]
    missing = [k for k in required_keys if k not in scalers]
    if missing:
        raise KeyError(f"scaler 檔案缺少欄位：{missing}")

    return scalers


def infer_lstm_hyperparams_from_state_dict(state_dict, default_hidden=128, default_layers=2):
    """
    從 PyTorch LSTM state_dict 推測 hidden_dim 與 num_layers。
    """
    hidden_dim = default_hidden
    num_layers = default_layers

    # lstm.weight_ih_l0 shape = (4*hidden_dim, input_dim)
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


def load_lstm_model(model_path: str, input_dim: int, output_dim: int, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")

    loaded = torch.load(model_path, map_location=device)

    # Case 1: torch.save(model, path)
    if isinstance(loaded, nn.Module):
        model = loaded
        model.to(device)
        model.eval()
        print("[Info] Loaded full PyTorch model object.")
        return model

    # Case 2: torch.save(model.state_dict(), path)
    if isinstance(loaded, dict):
        if "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        hidden_dim, num_layers = infer_lstm_hyperparams_from_state_dict(state_dict)
        print(f"[Info] Inferred LSTM hidden_dim={hidden_dim}, num_layers={num_layers}")

        model = LSTMWorldModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()

        print("[Info] Loaded model state_dict with LSTMWorldModel architecture.")
        return model

    raise TypeError(f"不支援的模型格式：{type(loaded)}")


def prepare_input_dataframe(raw_df: pd.DataFrame, input_columns: list):
    """
    將原始 CSV 轉成 scaler/model 需要的 input_columns。
    自動處理：
    - vehicle_type -> vehicle_type_EV / HEV / ICE
    - source_policy -> source_policy_DDPG / DDQN / DQN
    - 缺少欄位補 0
    """

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

    valid_mask = ~X_df.isna().any(axis=1)
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    cleaned_raw_df = raw_df.loc[valid_mask].reset_index(drop=True)

    return cleaned_raw_df, X_df


def load_dataset(data_path: str, input_columns: list):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到資料檔案：{data_path}")

    raw_df = pd.read_csv(data_path)

    if "sequence_id" not in raw_df.columns:
        raise KeyError("資料集缺少 sequence_id。LSTM SHAP 需要 sequence_id 來建立序列。")

    if "step" in raw_df.columns:
        raw_df = raw_df.sort_values(["sequence_id", "step"]).reset_index(drop=True)
    elif "episode" in raw_df.columns:
        raw_df = raw_df.sort_values(["sequence_id", "episode"]).reset_index(drop=True)
    else:
        raw_df = raw_df.sort_values(["sequence_id"]).reset_index(drop=True)

    cleaned_raw_df, X_df = prepare_input_dataframe(raw_df, input_columns)

    if len(X_df) == 0:
        raise ValueError("處理 one-hot 與缺值後，沒有可用資料。")

    return cleaned_raw_df, X_df


def build_sequence_windows(raw_df, X_df, x_scaler, input_columns, target_columns, seq_len, max_windows=None, seed=42):
    """
    依 sequence_id 建立 sliding windows。
    回傳：
      X_seq_scaled: (n_windows, seq_len, input_dim)
      meta_df: 每個 window 對應的 sequence_id / end_index / raw index
    """

    rng = np.random.default_rng(seed)

    rows = []
    meta = []

    # 將 X_df 接回 sequence_id 方便 group
    tmp = raw_df[["sequence_id"]].copy().reset_index(drop=True)
    tmp_X = X_df.reset_index(drop=True)
    tmp = pd.concat([tmp, tmp_X], axis=1)

    grouped = tmp.groupby("sequence_id", sort=False)

    for seq_id, group in grouped:
        group = group.reset_index(drop=True)

        if len(group) < seq_len:
            continue

        X_values = group[input_columns].values.astype(np.float32)
        X_scaled = x_scaler.transform(X_values).astype(np.float32)

        for end_pos in range(seq_len, len(group) + 1):
            window = X_scaled[end_pos - seq_len:end_pos]
            rows.append(window)
            meta.append({
                "sequence_id": seq_id,
                "window_end_pos": end_pos - 1,
                "window_start_pos": end_pos - seq_len,
            })

    if not rows:
        raise ValueError("沒有足夠長度的 sequence 可建立 LSTM windows。")

    X_seq = np.stack(rows, axis=0)
    meta_df = pd.DataFrame(meta)

    if max_windows is not None and len(X_seq) > max_windows:
        idx = rng.choice(len(X_seq), size=max_windows, replace=False)
        X_seq = X_seq[idx]
        meta_df = meta_df.iloc[idx].reset_index(drop=True)

    return X_seq, meta_df


def build_background_and_samples(X_seq, meta_df, background_size, sample_size, seed):
    rng = np.random.default_rng(seed)
    n = len(X_seq)

    background_size = min(background_size, n)
    sample_size = min(sample_size, n)

    bg_idx = rng.choice(n, size=background_size, replace=False)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_bg = X_seq[bg_idx]
    X_sample = X_seq[sample_idx]

    meta_bg = meta_df.iloc[bg_idx].reset_index(drop=True)
    meta_sample = meta_df.iloc[sample_idx].reset_index(drop=True)

    return X_bg, X_sample, meta_bg, meta_sample


def flatten_sequences(X_seq):
    """
    (n, seq_len, input_dim) -> (n, seq_len * input_dim)
    """
    n, seq_len, input_dim = X_seq.shape
    return X_seq.reshape(n, seq_len * input_dim)


def unflatten_sequences(X_flat, seq_len, input_dim):
    """
    (n, seq_len * input_dim) -> (n, seq_len, input_dim)
    """
    return X_flat.reshape(X_flat.shape[0], seq_len, input_dim)


def build_predict_fn(model, device, seq_len, input_dim):
    def predict_fn(x_flat_numpy):
        x_seq = unflatten_sequences(x_flat_numpy.astype(np.float32), seq_len, input_dim)
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            y = model(x_tensor).detach().cpu().numpy()

        return y

    return predict_fn


def normalize_shap_values(shap_values):
    """
    轉為 shape: (n_outputs, n_samples, n_flat_features)
    """
    if isinstance(shap_values, list):
        return np.array(shap_values)

    arr = np.array(shap_values)

    if arr.ndim == 3:
        # shap 新版常見：(n_samples, n_flat_features, n_outputs)
        return np.transpose(arr, (2, 0, 1))

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    raise ValueError(f"無法辨識 SHAP shape：{arr.shape}")


def aggregate_flat_shap_to_feature_level(shap_arr, seq_len, input_columns, aggregation="sum"):
    """
    shap_arr: (n_outputs, n_samples, seq_len * input_dim)
    return:
      feature_shap: (n_outputs, n_samples, input_dim)
    聚合各 timestep 的同一 feature。
    """

    n_outputs, n_samples, n_flat = shap_arr.shape
    input_dim = len(input_columns)

    expected_flat = seq_len * input_dim
    if n_flat != expected_flat:
        raise ValueError(f"SHAP flat feature 數不符：got {n_flat}, expected {expected_flat}")

    reshaped = shap_arr.reshape(n_outputs, n_samples, seq_len, input_dim)

    if aggregation == "sum":
        feature_shap = reshaped.sum(axis=2)
    elif aggregation == "mean":
        feature_shap = reshaped.mean(axis=2)
    else:
        raise ValueError("aggregation must be sum or mean")

    return feature_shap


def save_importance_outputs(
    feature_shap,
    input_columns,
    target_columns,
    output_dir,
    output_prefix,
):
    """
    feature_shap: (n_outputs, n_samples, input_dim)
    """

    os.makedirs(output_dir, exist_ok=True)

    global_importance = np.mean(np.abs(feature_shap), axis=(0, 1))

    global_df = pd.DataFrame({
        "feature": input_columns,
        "mean_abs_shap_all_targets": global_importance,
    }).sort_values("mean_abs_shap_all_targets", ascending=False)

    global_csv = os.path.join(output_dir, f"{output_prefix}_global_feature_importance.csv")
    global_df.to_csv(global_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = global_df.sort_values("mean_abs_shap_all_targets", ascending=True)
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap_all_targets"])
    ax.set_title(f"{output_prefix} Global SHAP Importance - All Targets", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    global_png = os.path.join(output_dir, f"{output_prefix}_global_feature_importance.png")
    plt.savefig(global_png, dpi=180)
    plt.close(fig)

    print(f"[Saved] {global_csv}")
    print(f"[Saved] {global_png}")

    for out_idx, target in enumerate(target_columns):
        if out_idx >= feature_shap.shape[0]:
            break

        imp = np.mean(np.abs(feature_shap[out_idx]), axis=0)

        target_df = pd.DataFrame({
            "feature": input_columns,
            "mean_abs_shap": imp,
        }).sort_values("mean_abs_shap", ascending=False)

        safe_target = target.replace("/", "_").replace("\\", "_").replace(" ", "_")
        target_csv = os.path.join(output_dir, f"{output_prefix}_{safe_target}_feature_importance.csv")
        target_df.to_csv(target_csv, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_df = target_df.sort_values("mean_abs_shap", ascending=True)
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
        ax.set_title(f"{output_prefix} SHAP Importance - {target}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()

        target_png = os.path.join(output_dir, f"{output_prefix}_{safe_target}_feature_importance.png")
        plt.savefig(target_png, dpi=180)
        plt.close(fig)


def save_raw_pickle(
    shap_arr_flat,
    feature_shap,
    expected_values,
    input_columns,
    target_columns,
    seq_len,
    sample_meta,
    output_dir,
    output_prefix,
):
    out_path = os.path.join(output_dir, f"{output_prefix}_shap_values.pkl")

    result = {
        "shap_values_flat": shap_arr_flat,
        "shap_values_feature_level": feature_shap,
        "expected_values": expected_values,
        "input_columns": input_columns,
        "target_columns": target_columns,
        "seq_len": seq_len,
        "sample_meta": sample_meta,
        "flat_format": "(n_outputs, n_samples, seq_len * n_features)",
        "feature_level_format": "(n_outputs, n_samples, n_features)",
        "aggregation": "sum over timesteps",
        "note": "Offline LSTM SHAP values generated by run_lstm_shap_offline.py",
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved] {out_path}")


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm_v1", "lstm_v2", "both"],
        default="lstm_v2",
        help="選擇要跑 SHAP 的 LSTM 模型。",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=10,
        help="Kernel SHAP background sequence window 數量。LSTM 建議先小。",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="要解釋的 sequence window 數量。LSTM 建議先小。",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=50,
        help="Kernel SHAP 每筆估計 perturbation 次數。越大越慢。",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=5000,
        help="最多先抽多少 sequence windows 供 SHAP 再抽樣，避免記憶體過大。",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["sum", "mean"],
        default="sum",
        help="將 seq_len 個 timestep 的同一 feature 聚合成 feature-level SHAP 的方式。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="shap_outputs",
        help="輸出資料夾。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu 或 cuda。",
    )

    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        raise ImportError("請先安裝 shap：pip install shap")

    os.makedirs(args.output_dir, exist_ok=True)

    model_keys = ["lstm_v1", "lstm_v2"] if args.model == "both" else [args.model]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]

        print("\n" + "=" * 80)
        print(f"Start LSTM SHAP: {cfg['display_name']}")
        print("=" * 80)

        scalers = load_scalers(cfg["scaler_path"])
        x_scaler = scalers["x_scaler"]
        input_columns = list(scalers["input_columns"])
        target_columns = list(scalers["target_columns"])

        seq_len = int(scalers.get("seq_len", 10))
        input_dim = len(input_columns)
        output_dim = len(target_columns)

        print(f"[Info] seq_len: {seq_len}")
        print(f"[Info] input_dim: {input_dim}")
        print(f"[Info] output_dim: {output_dim}")
        print(f"[Info] flat_dim for SHAP: {seq_len * input_dim}")
        print(f"[Info] input_columns: {input_columns}")
        print(f"[Info] target_columns: {target_columns}")

        raw_df, X_df = load_dataset(cfg["data_path"], input_columns)
        print(f"[Info] Dataset rows after preprocessing: {len(X_df)}")

        print("[Info] Building sequence windows...")
        X_seq, meta_df = build_sequence_windows(
            raw_df=raw_df,
            X_df=X_df,
            x_scaler=x_scaler,
            input_columns=input_columns,
            target_columns=target_columns,
            seq_len=seq_len,
            max_windows=args.max_windows,
            seed=args.seed,
        )

        print(f"[Info] Sequence windows available: {len(X_seq)}")

        X_bg, X_sample, meta_bg, meta_sample = build_background_and_samples(
            X_seq=X_seq,
            meta_df=meta_df,
            background_size=args.background,
            sample_size=args.samples,
            seed=args.seed,
        )

        print(f"[Info] Background windows: {len(X_bg)}")
        print(f"[Info] Sample windows: {len(X_sample)}")

        X_bg_flat = flatten_sequences(X_bg)
        X_sample_flat = flatten_sequences(X_sample)

        model = load_lstm_model(
            model_path=cfg["model_path"],
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
        )

        predict_fn = build_predict_fn(
            model=model,
            device=device,
            seq_len=seq_len,
            input_dim=input_dim,
        )

        print("[Info] Building KernelExplainer...")
        explainer = shap.KernelExplainer(predict_fn, X_bg_flat)

        print("[Info] Computing SHAP values...")
        print(
            f"[Info] background={len(X_bg_flat)}, samples={len(X_sample_flat)}, "
            f"nsamples={args.nsamples}, flat_features={X_bg_flat.shape[1]}"
        )

        shap_values = explainer.shap_values(X_sample_flat, nsamples=args.nsamples)

        shap_arr_flat = normalize_shap_values(shap_values)
        print(f"[Info] Flat SHAP shape: {shap_arr_flat.shape}")

        feature_shap = aggregate_flat_shap_to_feature_level(
            shap_arr=shap_arr_flat,
            seq_len=seq_len,
            input_columns=input_columns,
            aggregation=args.aggregation,
        )

        print(f"[Info] Feature-level SHAP shape: {feature_shap.shape}")

        save_raw_pickle(
            shap_arr_flat=shap_arr_flat,
            feature_shap=feature_shap,
            expected_values=explainer.expected_value,
            input_columns=input_columns,
            target_columns=target_columns,
            seq_len=seq_len,
            sample_meta=meta_sample,
            output_dir=args.output_dir,
            output_prefix=cfg["output_prefix"],
        )

        save_importance_outputs(
            feature_shap=feature_shap,
            input_columns=input_columns,
            target_columns=target_columns,
            output_dir=args.output_dir,
            output_prefix=cfg["output_prefix"],
        )

        print(f"[Done] LSTM SHAP completed for {model_key}")

    print("\n" + "=" * 80)
    print("All LSTM SHAP jobs completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
