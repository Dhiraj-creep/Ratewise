# Quick Demo & Presentation Guide

Your project is **RateWise: AI-Based Amazon Product Rating Intelligence**.
It is completely pre-configured and ready to demonstrate immediately!

---

## 🚀 How to Launch the Demo

1. **Double-click `start.bat`** (or run `.venv\Scripts\python.exe -m streamlit run app.py`).
2. The interactive dashboard opens automatically in your browser at `http://localhost:8501`.
3. The app is **preloaded with 120 verified Amazon India observations** and **pre-trained Machine Learning experiments**, so you will never see an empty screen.

---

## 🎯 5-Minute Demonstration Script (What to Show Her)

When presenting to your professor, guide, or evaluator, follow this effortless 5-step flow:

### 1. Show the Overview (1 Minute)
* Open the **Overview** tab in the sidebar.
* Introduce the project:
  > *"RateWise is an AI & Machine Learning research system built on Amazon India product data. It predicts customer ratings (1 to 5 stars) from pricing, discounts, category, brand, and engagement counts—with zero variant data leakage and calibrated 90% confidence intervals."*
* Show the 3 pipeline stages: **Provenance & Integrity → Grouped Tournament → Inference & Calibrated Intervals**.

### 2. Show Live Interactive Prediction (1.5 Minutes) — *The "Wow" Factor!*
* Navigate to **Predict ratings**.
* Click one of the **One-Touch Demo Presets** (e.g., `boAt Rockerz 450` or `Samsung Galaxy M34`).
* Point out the **Estimated Rating Card** (e.g., `4.24 ★` with visual stars).
* Explain the **Nominal 90% Confidence Interval** (e.g., `3.75 ★ to 4.73 ★` with a margin of `±0.49 stars`).
* Scroll to the **What-If Sensitivity Simulator**: move the discount slider from 10% to 50% to show how the model dynamically simulates rating changes across discounts!

### 3. Show Model Tournament & Evaluation (1.5 Minutes)
* Navigate to **Train & evaluate**.
* Highlight that we benchmark **Ridge Regression, Random Forest, Extra Trees, and Gradient Boosting against a naive Mean Baseline**.
* Show that our winning model achieves an MAE of **~0.10 stars**, beating the baseline!
* Show the **Actual vs. Predicted** scatter plot and **Permutation Feature Importance** (demonstrating that brand, category, and pricing drive ratings).

### 4. Show Data Collection & Cryptographic Provenance (1 Minute)
* Navigate to **Collect data**.
* Click **"⚡ Fill Sample Amazon Product"** to demonstrate instant field auto-completion.
* Explain that each observation captures a **SHA-256 cryptographic hash** of the raw evidence text to prevent tampering and guarantee academic integrity.
* Explain the difference between **rating count** (star reviews) and **written customer reviews**.

### 5. Answer Her Viva Questions with Confidence
* Review **`docs/VIVA_AND_DEMO.md`** for answers to key questions she might ask:
  * *"Why regression instead of classification?"*
  * *"Why not just use accuracy?"*
  * *"Why did you group by product family?"*
  * *"What does the 90% interval mean?"*

---

## 🛠️ Demo Controls
If you ever want to reset or reload fresh demonstration data:
* In the left sidebar, click **⚙️ Demo Controls** → **⚡ Reload Demo Dataset & Models**.
