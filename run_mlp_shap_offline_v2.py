# run_mlp_shap_offline_v2.py
# ============================================================
# Offline SHAP for MLP World Models
# 修正版：自動處理 vehicle_type / source_policy one-hot encoding
#
# 執行：
#   python run_mlp_shap_offline_v2.py --model mlp_v2 --background 30 --samples 80 --nsamples 100
#   python run_mlp_shap_offline_v2.py --model both --background 30 --samples 80 --nsamples 100
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


MODEL_CONFIGS = {
    "mlp_v1": {
        "display_name": "MLP v1 - Static World Model",
        "model_path": "world_model_mlp.pt",
        "scaler_path": "world_model_scalers.pkl",
        "data_path": "world_model_all.csv",
        "output_prefix": "mlp_v1",
    },
    "mlp_v2": {
        "display_name": "MLP v2 - Temporal World Model",
        "model_path": "world_model_mlp_sequence.pt",
        "scaler_path": "world_model_mlp_sequence_scalers.pkl",
        "data_path": "world_model_all_sequence.csv",
        "output_prefix": "mlp_v2",
    },
}


def load_scalers(scaler_path: str):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 scaler 檔案：{scaler_path}")

    scalers = joblib.load(scaler_path)

    required_keys = ["x_scaler", "y_scaler", "input_columns", "target_columns"]
    missing = [k for k in required_keys if k not in scalers]
    if missing:
        raise KeyError(f"scaler 檔案缺少欄位：{missing}")

    return scalers


def prepare_input_dataframe(raw_df: pd.DataFrame, input_columns: list):
    """
    將原始 CSV 轉成模型訓練時使用的 input_columns。
    主要處理：
    - vehicle_type -> vehicle_type_EV / HEV / ICE
    - source_policy -> source_policy_DDPG / DDQN / DQN
    - 若欄位不存在，自動補 0
    - 最後依照 scaler input_columns 排序
    """

    df = raw_df.copy()

    # 先保留原始欄位方便檢查
    original_columns = set(df.columns)

    # vehicle_type one-hot
    if "vehicle_type" in df.columns:
        dummies = pd.get_dummies(df["vehicle_type"], prefix="vehicle_type")
        df = pd.concat([df, dummies], axis=1)

    # source_policy one-hot
    if "source_policy" in df.columns:
        dummies = pd.get_dummies(df["source_policy"], prefix="source_policy")
        df = pd.concat([df, dummies], axis=1)

    # 有些資料可能是小寫或有空白，做一次保險清理
    if "vehicle_type" in original_columns:
        vehicle_clean = df["vehicle_type"].astype(str).str.strip().str.upper()
        for v in ["EV", "HEV", "ICE"]:
            col = f"vehicle_type_{v}"
            if col in input_columns and col not in df.columns:
                df[col] = (vehicle_clean == v).astype(int)

    if "source_policy" in original_columns:
        policy_clean = df["source_policy"].astype(str).str.strip().str.upper()
        for p in ["DDPG", "DDQN", "DQN"]:
            col = f"source_policy_{p}"
            if col in input_columns and col not in df.columns:
                df[col] = (policy_clean == p).astype(int)

    # 若 scaler 需要的欄位仍不存在，補 0
    # 這對 one-hot 很重要，例如資料剛好 sample 中沒有某類車，也要有該欄
    for col in input_columns:
        if col not in df.columns:
            df[col] = 0

    # 只取模型需要的欄位並轉 float
    X_df = df[input_columns].copy()

    # bool -> int
    for col in X_df.columns:
        if X_df[col].dtype == bool:
            X_df[col] = X_df[col].astype(int)

    # 轉數值，不能轉的變 NaN
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # 移除缺值
    valid_mask = ~X_df.isna().any(axis=1)
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    cleaned_raw_df = raw_df.loc[valid_mask].reset_index(drop=True)

    return cleaned_raw_df, X_df


def load_dataset(data_path: str, input_columns: list):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到資料檔案：{data_path}")

    raw_df = pd.read_csv(data_path)

    cleaned_raw_df, X_df = prepare_input_dataframe(raw_df, input_columns)

    if len(X_df) == 0:
        raise ValueError("處理 one-hot 與缺值後，沒有可用資料。")

    missing_after_prepare = [c for c in input_columns if c not in X_df.columns]
    if missing_after_prepare:
        raise KeyError(f"處理後仍缺少 input columns：{missing_after_prepare}")

    return cleaned_raw_df, X_df


def load_mlp_model(model_path: str, input_dim: int, output_dim: int, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")

    loaded = torch.load(model_path, map_location=device)

    if isinstance(loaded, nn.Module):
        model = loaded
        model.to(device)
        model.eval()
        print("[Info] Loaded full PyTorch model object.")
        return model

    if isinstance(loaded, dict):
        model = MLPWorldModel(input_dim=input_dim, output_dim=output_dim).to(device)

        if "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        model.load_state_dict(state_dict)
        model.eval()
        print("[Info] Loaded model state_dict.")
        return model

    raise TypeError(f"不支援的模型格式：{type(loaded)}")


def make_background_and_samples(
    raw_df: pd.DataFrame,
    X_df: pd.DataFrame,
    x_scaler,
    background_size: int,
    sample_size: int,
    seed: int,
):
    n = len(X_df)
    background_size = min(background_size, n)
    sample_size = min(sample_size, n)

    bg_idx = X_df.sample(n=background_size, random_state=seed).index
    sample_idx = X_df.sample(n=sample_size, random_state=seed + 1).index

    background_raw = raw_df.loc[bg_idx].reset_index(drop=True)
    sample_raw = raw_df.loc[sample_idx].reset_index(drop=True)

    X_bg_raw = X_df.loc[bg_idx].values.astype(np.float32)
    X_sample_raw = X_df.loc[sample_idx].values.astype(np.float32)

    X_bg_scaled = x_scaler.transform(X_bg_raw).astype(np.float32)
    X_sample_scaled = x_scaler.transform(X_sample_raw).astype(np.float32)

    return background_raw, sample_raw, X_bg_scaled, X_sample_scaled


def build_predict_fn(model, device: torch.device):
    def predict_fn(x_numpy):
        model.eval()
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
        with torch.no_grad():
            y = model(x_tensor).detach().cpu().numpy()
        return y

    return predict_fn


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return np.array(shap_values)

    arr = np.array(shap_values)

    if arr.ndim == 3:
        return np.transpose(arr, (2, 0, 1))

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    raise ValueError(f"無法辨識 SHAP shape：{arr.shape}")


def save_global_importance(
    shap_arr: np.ndarray,
    input_columns: list,
    target_columns: list,
    output_dir: str,
    output_prefix: str,
):
    global_importance = np.mean(np.abs(shap_arr), axis=(0, 1))

    global_df = pd.DataFrame({
        "feature": input_columns,
        "mean_abs_shap_all_targets": global_importance,
    }).sort_values("mean_abs_shap_all_targets", ascending=False)

    csv_path = os.path.join(output_dir, f"{output_prefix}_global_feature_importance.csv")
    global_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = global_df.sort_values("mean_abs_shap_all_targets", ascending=True)
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap_all_targets"])
    ax.set_title(f"{output_prefix} Global SHAP Importance - All Targets", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{output_prefix}_global_feature_importance.png")
    plt.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {png_path}")

    for out_idx, target in enumerate(target_columns):
        if out_idx >= shap_arr.shape[0]:
            break

        importance = np.mean(np.abs(shap_arr[out_idx]), axis=0)

        target_df = pd.DataFrame({
            "feature": input_columns,
            "mean_abs_shap": importance,
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


def save_raw_shap_pickle(
    shap_arr,
    expected_values,
    input_columns,
    target_columns,
    sample_raw_df,
    sample_scaled,
    output_dir,
    output_prefix,
):
    output_path = os.path.join(output_dir, f"{output_prefix}_shap_values.pkl")

    result = {
        "shap_values": shap_arr,
        "expected_values": expected_values,
        "input_columns": input_columns,
        "target_columns": target_columns,
        "sample_raw_df": sample_raw_df.reset_index(drop=True),
        "sample_scaled": sample_scaled,
        "format": "(n_outputs, n_samples, n_features)",
        "note": "Offline SHAP values generated by run_mlp_shap_offline_v2.py",
    }

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mlp_v1", "mlp_v2", "both"], default="mlp_v2")
    parser.add_argument("--background", type=int, default=30)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="shap_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        raise ImportError("請先安裝 shap：pip install shap")

    os.makedirs(args.output_dir, exist_ok=True)

    model_keys = ["mlp_v1", "mlp_v2"] if args.model == "both" else [args.model]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]

        print("\n" + "=" * 70)
        print(f"Start SHAP: {cfg['display_name']}")
        print("=" * 70)

        scalers = load_scalers(cfg["scaler_path"])
        x_scaler = scalers["x_scaler"]
        input_columns = list(scalers["input_columns"])
        target_columns = list(scalers["target_columns"])

        input_dim = len(input_columns)
        output_dim = len(target_columns)

        print(f"[Info] Input dim: {input_dim}")
        print(f"[Info] Output dim: {output_dim}")
        print(f"[Info] Input columns: {input_columns}")
        print(f"[Info] Target columns: {target_columns}")

        raw_df, X_df = load_dataset(cfg["data_path"], input_columns)
        print(f"[Info] Dataset rows after preprocessing: {len(X_df)}")

        background_raw, sample_raw, X_bg, X_sample = make_background_and_samples(
            raw_df=raw_df,
            X_df=X_df,
            x_scaler=x_scaler,
            background_size=args.background,
            sample_size=args.samples,
            seed=args.seed,
        )

        model = load_mlp_model(
            model_path=cfg["model_path"],
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
        )

        predict_fn = build_predict_fn(model, device)

        print("[Info] Building KernelExplainer...")
        explainer = shap.KernelExplainer(predict_fn, X_bg)

        print("[Info] Computing SHAP values...")
        print(f"[Info] background={len(X_bg)}, samples={len(X_sample)}, nsamples={args.nsamples}")
        shap_values = explainer.shap_values(X_sample, nsamples=args.nsamples)

        shap_arr = normalize_shap_values(shap_values)
        print(f"[Info] Normalized SHAP shape: {shap_arr.shape}")

        save_raw_shap_pickle(
            shap_arr=shap_arr,
            expected_values=explainer.expected_value,
            input_columns=input_columns,
            target_columns=target_columns,
            sample_raw_df=sample_raw,
            sample_scaled=X_sample,
            output_dir=args.output_dir,
            output_prefix=cfg["output_prefix"],
        )

        save_global_importance(
            shap_arr=shap_arr,
            input_columns=input_columns,
            target_columns=target_columns,
            output_dir=args.output_dir,
            output_prefix=cfg["output_prefix"],
        )

        print(f"[Done] SHAP completed for {model_key}")

    print("\n" + "=" * 70)
    print("All SHAP jobs completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
