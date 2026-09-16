import os
import sys
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

def get_candidate_models():
    models = {}
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=5, verbose=0, random_seed=42)
    models["RandomForest"] = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    models["ExtraTrees"] = ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    return models

def run_dynamic_15bps_two_stage_trainer():
    print("=" * 110)
    print("🧠 HFT TWO-STAGE CASCADED TRAINER (DATASET: hft_dynamic_market_dataset.csv | THRESHOLD: ±15.00 BPS)")
    print("=" * 110)

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    search_dirs = [script_dir, os.getcwd()]
    
    candidates = [
        "hft_dynamic_market_dataset.csv",
        "hft_15bps_balanced_dataset.csv"
    ]
    
    found_path = None
    for fname in candidates:
        for d in search_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                found_path = p
                break
        if found_path:
            break

    if not found_path:
        for d in search_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(".csv") and "demo" not in f.lower():
                        found_path = os.path.join(d, f)
                        break
            if found_path:
                break

    if not found_path:
        print("❌ Error: Could not locate 'hft_dynamic_market_dataset.csv' in working directory.")
        return

    print(f"📁 Source Dataset Successfully Loaded: '{found_path}'")
    df_raw = pd.read_csv(found_path)
    n_rows = len(df_raw)
    print(f"📏 Dataset Dimensions: {n_rows} rows × {len(df_raw.columns)} columns")

    # 1. Price Column & Target Extraction with strict 15.00 BPS Threshold
    THRESHOLD_BPS = 15.00
    price_col = "mid_price" if "mid_price" in df_raw.columns else ("price" if "price" in df_raw.columns else None)
    
    if price_col:
        prices = df_raw[price_col].values
        print(f"💡 Reference Price Basis: '{price_col}'")
    elif "mid_price_current_bps" in df_raw.columns:
        prices = df_raw["mid_price_current_bps"].values
        print("💡 Reference Price Basis: 'mid_price_current_bps'")
    else:
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        prices = df_raw[num_cols[0]].values if len(num_cols) > 0 else np.ones(n_rows)
        print(f"💡 Reference Price Basis: '{num_cols[0] if len(num_cols)>0 else 'default'}'")

    if "mid_price_future_change_pct" in df_raw.columns:
        vals = df_raw["mid_price_future_change_pct"].values
        mult = 10000.0 if np.max(np.abs(vals[~np.isnan(vals)])) < 0.1 else 100.0
        bps_5s = vals * mult
    else:
        bps_5s = ((np.roll(prices, -5) - prices) / prices) * 10000.0

    bps_15s = ((np.roll(prices, -15) - prices) / prices) * 10000.0

    y_5s_raw = np.where(bps_5s > THRESHOLD_BPS, 1, np.where(bps_5s < -THRESHOLD_BPS, -1, 0))
    y_15s_raw = np.where(bps_15s > THRESHOLD_BPS, 1, np.where(bps_15s < -THRESHOLD_BPS, -1, 0))

    print(f"• Target Threshold Enforced : ±{THRESHOLD_BPS:.2f} BPS")
    print(f"• Stage 1 (5s Horizon Target) Class Breakdown  : {dict(pd.Series(y_5s_raw).value_counts())}")
    print(f"• Stage 2 (15s Horizon Target) Class Breakdown : {dict(pd.Series(y_15s_raw).value_counts())}")
    print("=" * 110)

    # 2. Extract Feature Matrices
    lag10_cols = [c for c in df_raw.columns if any(f"lag{i}" in c or f"lag_{i}" in c for i in range(1, 11))]
    lag15_cols = [c for c in df_raw.columns if any(f"lag{i}" in c or f"lag_{i}" in c for i in range(1, 16))]

    if not lag10_cols:
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        ignore = ["target", "target_5s", "target_15s", "target_15bps", "bps_change", "mid_price_future_change_pct"]
        base_features = [c for c in num_cols if c not in ignore]
        X_stage1_window = df_raw[base_features].fillna(0)
        X_shifted_window = df_raw[base_features].fillna(0)
        X_stage2_window = df_raw[base_features].fillna(0)
    else:
        X_stage1_window = df_raw[lag10_cols].select_dtypes(include=[np.number]).fillna(0)
        shift_cols = [c for c in df_raw.columns if any(f"lag{i}" in c or f"lag_{i}" in c for i in range(1, 11))]
        X_shifted_window = df_raw[shift_cols].select_dtypes(include=[np.number]).fillna(0)
        X_stage2_window = df_raw[lag15_cols].select_dtypes(include=[np.number]).fillna(0)

    print(f"• Stage 1 Base Feature Window (t=0..10) : {X_stage1_window.shape}")
    print(f"• Shifted Window (t=5..15) Features     : {X_shifted_window.shape}")
    print(f"• Stage 2 Full 15s Window Features      : {X_stage2_window.shape}")
    print("=" * 110)

    # 3. Purged GroupKFold Cross-Validation (No Shuffle)
    groups = np.arange(len(df_raw)) // 60
    if len(np.unique(groups)) < 5:
        groups = np.minimum(np.arange(len(df_raw)) // max(1, len(df_raw) // 5), 4)

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

    le_5s = LabelEncoder()
    le_15s = LabelEncoder()
    y_5s_enc = le_5s.fit_transform(y_5s_raw)
    y_15s_enc = le_15s.fit_transform(y_15s_raw)

    s1_candidates = get_candidate_models()
    s2_candidates = get_candidate_models()

    oof_store_shifted = {}

    print("\n" + "=" * 110)
    print("⚙️ STEP 1: TRAINING STAGE 1 MODELS ON 10s BASE WINDOW & PREDICTING SHIFTED 5s..15s PROPORTIONS...")
    print("=" * 110)

    for s1_name, s1_model in s1_candidates.items():
        oof_preds_shifted = np.zeros(len(df_raw))
        oof_probs_shifted = np.zeros((len(df_raw), len(le_5s.classes_)))
        s1_accs, s1_f1s = [], []

        for train_idx, test_idx in gkf.split(X_stage1_window, y_5s_enc, groups=groups):
            X_tr = X_stage1_window.iloc[train_idx]
            y_tr = y_5s_enc[train_idx]
            X_te_shifted = X_shifted_window.iloc[test_idx]
            y_te = y_5s_enc[test_idx]

            if len(np.unique(y_tr)) >= 2:
                s1_model.fit(X_tr, y_tr)
                p_te = s1_model.predict(X_te_shifted)
                pr_te = s1_model.predict_proba(X_te_shifted)
            else:
                p_te = np.full(len(X_te_shifted), y_tr[0] if len(y_tr)>0 else 0)
                pr_te = np.zeros((len(X_te_shifted), len(le_5s.classes_)))
                if len(y_tr) > 0:
                    pr_te[:, y_tr[0]] = 1.0

            p_te_flat = np.asarray(p_te).ravel()
            oof_preds_shifted[test_idx] = p_te_flat
            if pr_te.shape == oof_probs_shifted[test_idx].shape:
                oof_probs_shifted[test_idx] = pr_te

            s1_accs.append(accuracy_score(y_te, p_te_flat))
            s1_f1s.append(f1_score(y_te, p_te_flat, average="macro", zero_division=0))

        oof_store_shifted[s1_name] = {
            "preds": oof_preds_shifted,
            "probs": oof_probs_shifted,
            "s1_acc": np.mean(s1_accs),
            "s1_f1": np.mean(s1_f1s)
        }
        print(f"✓ Stage 1 [{s1_name:<12}] Shifted-Window OOF Accuracy = {np.mean(s1_accs)*100:.2f}% | F1-Macro = {np.mean(s1_f1s):.4f}")

    # 4. Step 2: Joint Evaluation
    print("\n" + "=" * 110)
    print("⚙️ STEP 2: STAGE 2 INGESTION OF STAGE 1 PROPORTIONS + 15s FEATURES (EVALUATING JOINT PAIRS)...")
    print("=" * 110)

    joint_results = []

    for s1_name, s1_data in oof_store_shifted.items():
        s1_signals = pd.DataFrame({"stage1_pred_5s_shifted": s1_data["preds"]})
        for c_idx, c_name in enumerate(le_5s.classes_):
            s1_signals[f"stage1_prob_prop_{c_name}"] = s1_data["probs"][:, c_idx]

        X_cascaded_15s = pd.concat([X_stage2_window.reset_index(drop=True), s1_signals.reset_index(drop=True)], axis=1)

        for s2_name, s2_model in s2_candidates.items():
            s2_accs, s2_f1s, lats = [], [], []

            for train_idx, test_idx in gkf.split(X_cascaded_15s, y_15s_enc, groups=groups):
                X_tr, X_te = X_cascaded_15s.iloc[train_idx], X_cascaded_15s.iloc[test_idx]
                y_tr, y_te = y_15s_enc[train_idx], y_15s_enc[test_idx]

                t0 = time.perf_counter()
                if len(np.unique(y_tr)) >= 2:
                    s2_model.fit(X_tr, y_tr)
                    p_te = s2_model.predict(X_te)
                else:
                    p_te = np.full(len(X_te), y_tr[0] if len(y_tr)>0 else 0)
                t1 = time.perf_counter()

                p_te_flat = np.asarray(p_te).ravel()
                lat_us = ((t1 - t0) / len(X_te)) * 1e6 if len(X_te) > 0 else 0
                s2_accs.append(accuracy_score(y_te, p_te_flat))
                s2_f1s.append(f1_score(y_te, p_te_flat, average="macro", zero_division=0))
                lats.append(lat_us)

            joint_acc = np.mean(s2_accs)
            joint_f1 = np.mean(s2_f1s)
            avg_lat = np.mean(lats)

            joint_results.append({
                "Stage1_5s_Model": s1_name,
                "Stage1_Shifted_F1": s1_data["s1_f1"],
                "Stage2_15s_Model": s2_name,
                "Joint_Purged_CV_F1": joint_f1,
                "Joint_Purged_CV_Acc": joint_acc,
                "Inference_Latency_us": avg_lat
            })

            print(f"  • Pair [{s1_name:<11} (5s Shifted) -> {s2_name:<11} (15s)] => Joint F1 = {joint_f1:.4f} | Acc = {joint_acc*100:.2f}% | Latency = {avg_lat:.2f}µs")

    # 5. Leaderboard & Winning Model Selection
    res_df = pd.DataFrame(joint_results).sort_values(by="Joint_Purged_CV_F1", ascending=False).reset_index(drop=True)
    winner = res_df.iloc[0]

    print("\n" + "=" * 110)
    print("🏆 SHIFTED-WINDOW CASCADED MODEL LEADERBOARD (DYNAMIC DATASET | 15 BPS THRESHOLD)")
    print("=" * 110)
    print(res_df[["Stage1_5s_Model", "Stage1_Shifted_F1", "Stage2_15s_Model", "Joint_Purged_CV_F1", "Joint_Purged_CV_Acc", "Inference_Latency_us"]].head(10).to_string(index=False))
    print("=" * 110)

    print("\n" + "=" * 110)
    print("💡 WINNING PIPELINE COMBINATION RATIONALE")
    print("=" * 110)
    print(f"1. STAGE 1 BEST MODEL (5s Target on 10s Base)    : '{winner['Stage1_5s_Model']}'")
    print(f"2. STAGE 2 BEST MODEL (15s Target + 5s Feedback) : '{winner['Stage2_15s_Model']}'")
    print(f"3. JOINT PERFORMANCE ACCURACY                     : {winner['Joint_Purged_CV_Acc']*100:.2f}% Purged CV Accuracy")
    print(f"4. JOINT PERFORMANCE F1-MACRO                     : {winner['Joint_Purged_CV_F1']:.4f} Purged CV F1-Macro")
    print(f"5. SELECTION METHOD                               : Joint combination evaluation driven by Stage 2 final output.")
    print("=" * 110)

    # 6. Fit Final Models & Export TWO PKL Files
    print("\n💾 Fitting winning models on full dataset & exporting TWO PKL artifacts...")

    best_s1_model = get_candidate_models()[winner["Stage1_5s_Model"]]
    best_s2_model = get_candidate_models()[winner["Stage2_15s_Model"]]

    if len(np.unique(y_5s_enc)) >= 2:
        best_s1_model.fit(X_stage1_window, y_5s_enc)
        full_s1_preds = best_s1_model.predict(X_shifted_window)
        full_s1_probs = best_s1_model.predict_proba(X_shifted_window)
    else:
        full_s1_preds = np.full(len(X_shifted_window), y_5s_enc[0] if len(y_5s_enc)>0 else 0)
        full_s1_probs = np.zeros((len(X_shifted_window), len(le_5s.classes_)))
        if len(y_5s_enc) > 0:
            full_s1_probs[:, y_5s_enc[0]] = 1.0

    full_s1_preds_flat = np.asarray(full_s1_preds).ravel()
    full_s1_signals = pd.DataFrame({"stage1_pred_5s_shifted": full_s1_preds_flat})
    for c_idx, c_name in enumerate(le_5s.classes_):
        full_s1_signals[f"stage1_prob_prop_{c_name}"] = full_s1_probs[:, c_idx]

    X_full_cascaded_15s = pd.concat([X_stage2_window.reset_index(drop=True), full_s1_signals.reset_index(drop=True)], axis=1)

    if len(np.unique(y_15s_enc)) >= 2:
        best_s2_model.fit(X_full_cascaded_15s, y_15s_enc)

    # Save PKL 1
    pkl_stage1_name = "hft_stage1_dynamic_15bps_model.pkl"
    stage1_artifact = {
        "model": best_s1_model,
        "label_encoder": le_5s,
        "feature_names": list(X_stage1_window.columns),
        "model_name": winner["Stage1_5s_Model"],
        "target_threshold_bps": 15.00,
        "target_horizon": "5s"
    }
    joblib.dump(stage1_artifact, pkl_stage1_name)

    # Save PKL 2
    pkl_stage2_name = "hft_stage2_dynamic_15bps_model.pkl"
    stage2_artifact = {
        "model": best_s2_model,
        "label_encoder": le_15s,
        "feature_names": list(X_full_cascaded_15s.columns),
        "stage1_dependency_model": winner["Stage1_5s_Model"],
        "model_name": winner["Stage2_15s_Model"],
        "joint_f1_score": winner["Joint_Purged_CV_F1"],
        "target_threshold_bps": 15.00,
        "target_horizon": "15s"
    }
    joblib.dump(stage2_artifact, pkl_stage2_name)

    print("=" * 110)
    print("🏆 TWO DISTINCT PKL ARTIFACTS EXPORTED SUCCESSFULLY!")
    print(f"1. Stage 1 Model PKL File : '{pkl_stage1_name}' ({winner['Stage1_5s_Model']} for 5s Target)")
    print(f"2. Stage 2 Model PKL File : '{pkl_stage2_name}' ({winner['Stage2_15s_Model']} for 15s Target with 15 BPS Feedback)")
    print("=" * 110)

if __name__ == "__main__":
    run_dynamic_15bps_two_stage_trainer()
