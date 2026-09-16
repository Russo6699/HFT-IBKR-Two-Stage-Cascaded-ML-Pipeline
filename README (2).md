# ⚡ High-Frequency Trading (HFT) Two-Stage Cascaded ML Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![IBKR API](https://img.shields.io/badge/Interactive%20Brokers-API-red.svg)](https://interactivebrokers.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-frequency quantitative trading (HFT) machine learning architecture designed to stream, process, and forecast stock price movements using real-time Limit Order Book (LOB) data from the Interactive Brokers (IBKR) API.

The system features real-time microstructure feature engineering, automated dynamic basis point (BPS) threshold discovery, and a **Shifted Two-Stage Cascaded ML Architecture** trained with **Purged GroupKFold Cross-Validation** to strictly eliminate temporal lookahead bias.

---

## 🏛️ System Architecture

```text
                                  LIVE IBKR DATA STREAM
                               (Level 1 & Level 2 Order Book)
                                             │
                                             ▼
                               [Microstructure Engineering]
                        (OFI, CVD, Layer Imbalances, Velocity, BPS)
                                             │
                                             ▼
                             [Dynamic BPS Threshold Search]
                         (Targeting ~20% Neutral Buffer @ ±24.50 BPS)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [Stage 1: 5s Auxiliary Model]              [10s Base Feature Window]
             (Predicts short-term shift)                           │
                       │                                           │
                       └──────────────► [Probability Injection] ──┘
                                             │
                                             ▼
                              [Stage 2: 15s Primary Model]
                             (Final Trade Decision: UP/DOWN/NEUTRAL)
                                             │
                                             ▼
                                [Model Artifact Serialization]
                                (hft_stage1.pkl, hft_stage2.pkl)
```

---

## ✨ Key Features & Technical Highlights

- **Real-Time Data Streaming:** Asynchronous ingestion (`ib_insync` / `asyncio`) of Tier-1 and Tier-2 tick data during US market open and close volatility windows.
- **Market Microstructure Engineering:** Computes 180+ lagged features across a rolling 15-second window:
  - **Order Flow Imbalance (OFI):** Quantifies net buying/selling pressure at top-of-book.
  - **Cumulative Volume Delta (CVD):** Measures net transactional volume direction.
  - **5-Layer Depth Imbalance:** Evaluates supply/demand pressure across 5 order book levels.
  - **Kinetic Metrics:** Real-time Price Velocity (`vel_bps`), Acceleration (`acc_bps`), and Spread BPS.
- **Automated Dynamic BPS Thresholding:** Automated grid-search algorithm discovering optimal target sensitivity (±24.50 BPS / 0.245%) for ideal 3-class balance:
  - **41.6% UP (`+1`)** | **38.5% DOWN (`-1`)** | **19.9% NEUTRAL (`0`)** (noise reduction buffer).
- **Shifted Two-Stage Cascaded ML Architecture:**
  - **Stage 1 (5s Horizon):** Trains on $t=0..10s$ base features to forecast immediate 5-second momentum shifts.
  - **Probability Vector Injection:** Injects Stage 1 out-of-fold class probability predictions into Stage 2 feature matrix.
  - **Stage 2 (15s Horizon):** Evaluates full 15s features augmented with Stage 1 probability signals for final execution decision.
- **Purged GroupKFold Cross-Validation:** Time-series validation framework ensuring no future information leakage or autocorrelation overlap between folds.

---

## 📊 Benchmark Leaderboard

Evaluated across candidate models (ExtraTrees, RandomForest, CatBoost, XGBoost, LightGBM):

| Rank | Pipeline Pair (Stage 1 -> Stage 2) | Purged CV Accuracy | F1-Macro Score | Latency Profile |
| :---: | :--- | :---: | :---: | :---: |
| **#1** | **ExtraTrees -> RandomForest** | **44.83%** | **0.3870** | **Low (<1.5ms)** |
| #2 | CatBoost -> CatBoost | 43.12% | 0.3795 | Medium (~5ms) |
| #3 | ExtraTrees -> LightGBM | 42.90% | 0.3710 | Ultra-Low (<1ms) |
| #4 | XGBoost -> XGBoost | 41.65% | 0.3620 | Low (~2ms) |

*Baseline random guess accuracy for 3 balanced classes is 33.33%.*

---

## 📁 Repository Structure

```text
├── data/
│   └── hft_dynamic_market_dataset_sample.csv   # Sample dataset (180 features + target BPS)
├── models/
│   ├── hft_stage1_unified_20pct_model.pkl       # Serialized Stage 1 model
│   └── hft_stage2_unified_20pct_model.pkl       # Serialized Stage 2 model
├── scripts/
│   ├── hft_data_collector.py                    # Live IBKR L1/L2 data stream collector
│   ├── hft_threshold_analyzer.py                # BPS threshold distribution analyzer
│   └── hft_cascaded_trainer.py                  # Two-Stage Cascaded Trainer & CV Pipeline
├── .gitignore                                   # Standard Python Git ignore rules
├── requirements.txt                             # Dependencies
└── README.md                                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

```bash
git clone https://github.com/YOUR_USERNAME/hft-cascaded-ml-pipeline.git
cd hft-cascaded-ml-pipeline
pip install -r requirements.txt
```

### 2. Run Data Collector (Requires IBKR TWS / Gateway running)

```bash
python scripts/hft_data_collector.py
```

### 3. Run Threshold Analyzer & Cascaded Trainer

```bash
python scripts/hft_cascaded_trainer.py
```

---

## 🛠️ Stack & Technologies

- **Language:** Python 3.12
- **Quantitative & ML Libraries:** Scikit-Learn, CatBoost, XGBoost, LightGBM, Pandas, NumPy
- **Broker Connectivity:** Interactive Brokers API (`ib_insync`, `nest_asyncio`)
- **Model Serialization:** Joblib

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
