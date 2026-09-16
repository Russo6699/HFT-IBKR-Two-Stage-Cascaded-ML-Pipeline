# 🚀 GitHub Project Setup Guide (HFT Cascaded ML Pipeline)

This step-by-step guide will walk you through setting up your High-Frequency Trading (HFT) Machine Learning repository on GitHub for your portfolio and resume.

---

## 📋 Step 1: Organize Your Local Project Directory (VS Code)

1. Open your project folder in VS Code (`C:\Users\Ruslan-PC\Desktop\LM\Project`).
2. Create a subdirectory named `scripts/` and move your Python scripts into it:
   - `hft_data_collector.py`
   - `hft_unified_20pct_neutral_cascaded_trainer_v2.py`
3. Ensure the following root setup files are located in your root project folder:
   - `README.md` (Main technical documentation)
   - `.gitignore` (Excludes large CSV datasets, cache, and binary artifacts)
   - `requirements.txt` (Python dependencies)

---

## 🌐 Step 2: Create a New Repository on GitHub

1. Sign in to your GitHub account at [github.com](https://github.com).
2. Click the **`New`** button (or the `+` icon at the top right -> **New repository**).
3. Fill in the repository details:
   - **Repository name:** `hft-cascaded-ml-pipeline`
   - **Description:** `High-Frequency Trading Two-Stage Cascaded ML Pipeline using IBKR Level 2 Data & Microstructure Engineering`
   - **Visibility:** Select **Public** (so recruiters and hiring managers can view it).
   - **Do NOT check** "Add a README file" (as we already created a customized `README.md`).
4. Click **`Create repository`**.

---

## 💻 Step 3: Link VS Code Terminal & Push Code to GitHub

Open the terminal in VS Code (`Ctrl + ~`) and run the following Git commands in sequence:

```bash
# 1. Initialize local Git repository
git init

# 2. Stage all files (respecting .gitignore)
git add .

# 3. Create your initial commit
git commit -m "Initial commit: HFT IBKR Two-Stage Cascaded ML Pipeline"

# 4. Set the default branch to main
git branch -M main

# 5. Link local repository to your remote GitHub repo (Replace with your actual GitHub URL!)
git remote add origin https://github.com/YOUR_USERNAME/hft-cascaded-ml-pipeline.git

# 6. Push your code to GitHub
git push -u origin main
```

---

## 🌟 Step 4: Profile & Resume Optimization Tips

1. **Add Repository Topics on GitHub:**  
   On your GitHub repository page, click the gear icon next to **About** and add relevant topics:  
   `hft`, `machine-learning`, `quantitative-finance`, `interactive-brokers`, `order-book`, `scikit-learn`, `python`.

2. **Resume & LinkedIn Inclusion:**  
   Under the **Projects** section of your resume or LinkedIn profile, add your repository link alongside a bullet summary:
   > **HFT Quantitative ML Pipeline | Python, Scikit-Learn, IBKR API, CatBoost**  
   > *Developed an end-to-end HFT ML system using real-time IBKR Level 2 data streaming and engineered Order Flow Imbalance (OFI) & CVD features. Implemented a Shifted Two-Stage Cascaded Pipeline with Purged GroupKFold CV, optimizing BPS thresholds for balanced 3-class trade execution.*
