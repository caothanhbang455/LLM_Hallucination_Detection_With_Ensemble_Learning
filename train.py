import os
import argparse
import json
import numpy as np
import pandas as pd
import sys

from src.config import cfg
from src.fold_train import run_kfold_training
from src.ensemble import stacked_predict

# Nếu đang chạy trong Jupyter Notebook, fake argv để tránh argparse lỗi
if "ipykernel" in sys.modules:
    sys.argv = [
        "train.py",
        "--mode", "predict",
        "--models_out_dir", "outputs/20250922_104022"  # thay bằng folder muốn predict
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "predict", "train_then_predict"], default="train_then_predict",
                    help="Chọn chế độ train, predict, hoặc train_then_predict")
parser.add_argument("--test_csv", default=None,
                    help="Ghi đè đường dẫn test csv (mặc định dùng cfg.test_csv)")
parser.add_argument("--models_out_dir", default=None,
                    help="Nếu chỉ predict, cần cung cấp thư mục chứa model đã train")
parser.add_argument("--save_preds", default="predictions.csv",
                    help="Tên file csv lưu kết quả dự đoán")
args = parser.parse_args()

out_dir = args.models_out_dir

# ===== TRAINING =====
if args.mode in ["train", "train_then_predict"]:
    out_dir = run_kfold_training()
    print(f"Artifacts at: {out_dir}")

# ===== PREDICTION =====
if args.mode in ["predict", "train_then_predict"]:
    test_csv = args.test_csv or cfg.test_csv
    if test_csv is None or not os.path.exists(test_csv):
        raise ValueError(f"Test CSV not found: {test_csv}")
    if out_dir is None or not os.path.exists(out_dir):
        raise ValueError("Missing models_out_dir to run prediction")

    preds, label_names = stacked_predict(test_csv, out_dir)

    df_test = pd.read_csv(test_csv)
    if "id" in df_test.columns:
        sub = pd.DataFrame({
            "id": df_test["id"].values,
            "label": [label_names[i] for i in preds]
        })
    else:
        sub = pd.DataFrame({"label": [label_names[i] for i in preds]})

    sub_path = os.path.join(out_dir, args.save_preds)
    sub.to_csv(sub_path, index=False)
    print(f"Saved predictions to {sub_path}")

    np.save(os.path.join(out_dir, "predictions.npy"), preds)
    with open(os.path.join(out_dir, "label_names.json"), "w") as f:
        json.dump({"label_names": label_names}, f, indent=2)

print(f"Saved preds npy and label names under {out_dir}")
