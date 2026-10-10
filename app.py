from datetime import datetime, timezone
from pathlib import Path
import io
import json
import os
import random
import sqlite3
import zipfile

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ratewise.collector import parse_saved_html
from ratewise.data import COLUMNS, load_observations, save_observation, safe_csv, validate_frame, number, FEATURES, NUMERIC
from ratewise.model import train, predict
from ratewise.demo_data import seed_demo_data, DEMO_PRODUCTS

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RATEWISE_DATA_DIR", str(ROOT / "data")))
RUNS = Path(os.environ.get("RATEWISE_RUNS_DIR", str(ROOT / "runs")))
CATEGORIES = ["Electronics", "Home & Kitchen", "Beauty & Personal Care", "Sports & Fitness", "Office Products"]

st.set_page_config(page_title="RateWise | Amazon Rating Intelligence", page_icon="◈", layout="wide")

# Custom CSS for modern, high-contrast, polished demo aesthetic
st.markdown("""<style>
.block-container { padding-top: 2rem; max-width: 1320px; }
h1 { letter-spacing: -1.5px !important; font-weight: 750 !important; }
h2, h3 { letter-spacing: -.5px !important; }
[data-testid="stMetric"] { background: #131f32; border: 1px solid #24344d; border-radius: 14px; padding: 18px; }
.eyebrow { color: #49d9c2; font-size: 12px; letter-spacing: 2.5px; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }
.hero { background: linear-gradient(120deg, #132738, #101c2e); border: 1px solid #23445d; border-radius: 18px; padding: 26px 30px; margin: 10px 0 24px; }
.hero h2 { margin: 0 0 6px; font-size: 30px; color: #ffffff; }
.hero p { color: #b7c8da; max-width: 820px; margin: 0; font-size: 15px; line-height: 1.5; }
.demo-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(73, 217, 194, 0.12); color: #49d9c2; border: 1px solid rgba(73, 217, 194, 0.35); border-radius: 999px; padding: 4px 14px; font-size: 12px; font-weight: 600; margin-bottom: 12px; }
.step-card { background: #131f32; border: 1px solid #22354c; border-radius: 12px; padding: 18px; height: 100%; }
.step-num { font-size: 11px; font-weight: 800; color: #49d9c2; letter-spacing: 2px; }
.step-title { font-size: 17px; font-weight: 700; color: #ffffff; margin: 4px 0 8px; }
.step-desc { font-size: 13.5px; color: #9bb1c9; line-height: 1.45; }
.rating-card { background: linear-gradient(135deg, #15293d, #142032); border: 1px solid #2f5070; border-radius: 18px; padding: 26px; text-align: center; margin: 16px 0; }
.rating-num { font-size: 52px; font-weight: 800; color: #ffffff; line-height: 1; }
.rating-stars { font-size: 26px; color: #f5b041; margin: 8px 0; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 700; }
.badge-green { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.3); }
.badge-blue { background: rgba(52, 152, 219, 0.15); color: #3498db; border: 1px solid rgba(52, 152, 219, 0.3); }
.badge-amber { background: rgba(243, 156, 18, 0.15); color: #f39c12; border: 1px solid rgba(243, 156, 18, 0.3); }
.qa-box { background: #121c2c; border-left: 4px solid #49d9c2; border-radius: 8px; padding: 14px 18px; margin: 10px 0; font-size: 14px; }
</style>""", unsafe_allow_html=True)

# Auto-seed demo data if observations database doesn't exist
df = load_observations(DATA)
if len(df) == 0:
    try:
        seed_demo_data(DATA, RUNS)
        df = load_observations(DATA)
    except Exception:
        pass

# Sidebar navigation
st.sidebar.markdown("## ◈ RateWise")
st.sidebar.caption("AMAZON INDIA RATING INTELLIGENCE")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Collect data", "Explore dataset", "Train & evaluate", "Predict ratings", "Project guide"]
)

st.sidebar.divider()
st.sidebar.caption("DATASET STATUS")
st.sidebar.metric("Collected observations", len(df))

# Sidebar Quick Actions
with st.sidebar.expander("⚙️ Demo Controls"):
    if st.button("⚡ Reload Demo Dataset & Models", use_container_width=True):
        with st.spinner("Seeding demo data and training models..."):
            seed_demo_data(DATA, RUNS, force=True)
            df = load_observations(DATA)
            st.success("Demo dataset & models reloaded!")
            st.rerun()

    if st.button("🧹 Clear All Data", use_container_width=True):
        if (DATA / "observations.sqlite").exists():
            (DATA / "observations.sqlite").unlink()
        st.warning("Observations cleared!")
        st.rerun()

def header(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)

def csv_download(frame, label="Download CSV", name="amazon_products.csv"):
    st.download_button(label, safe_csv(frame), name, "text/csv")

def get_runs():
    return sorted([x for x in RUNS.glob("*") if (x / "metrics.json").exists()], reverse=True) if RUNS.exists() else []

def run_picker(key):
    runs = get_runs()
    if not runs:
        st.info("No trained model yet. Collect genuine Amazon data or reload demo models from Demo Controls.")
        return None
    return st.selectbox("Selected Experiment Run", runs, format_func=lambda p: p.name, key=key)

def render_star_display(val):
    val = float(val)
    full = int(val)
    rem = val - full
    half = 1 if 0.25 <= rem < 0.75 else (0 if rem < 0.25 else 0)
    if rem >= 0.75:
        full += 1
    empty = max(0, 5 - full - half)
    return "★" * full + ("½" if half else "") + "☆" * empty

def show_results(path):
    r = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    
    # Header summary banner
    is_better = r.get("test_beats_mean_baseline", False)
    status_color = "#2ecc71" if is_better else "#f39c12"
    status_text = "MODEL OUTPERFORMS NAIVE BASELINE" if is_better else "BASELINE BENCHMARK COMPARISON"
    
    st.markdown(f'<div class="demo-pill" style="border-color:{status_color};color:{status_color};">● {status_text} · Winner: {r["winner"]}</div>', unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Test MAE", f"{r['test_metrics']['MAE']:.3f}", help="Mean Absolute Error on held-out test products. Lower is better.")
    b.metric("Test RMSE", f"{r['test_metrics']['RMSE']:.3f}", help="Root Mean Squared Error. Penalizes large misses heavily.")
    c.metric("Baseline MAE", f"{r['baseline_test_metrics']['MAE']:.3f}", help="Naive baseline predicting the training set average.")
    d.metric("Nominal 90% Coverage", f"{r['interval']['test_product_coverage']:.1%}", help="Percentage of test products that fell inside the calibrated interval.")

    st.markdown(f"""
    > **Plain English Takeaway:** The winning **{r['winner']}** achieves an average estimation error of **±{r['test_metrics']['MAE']:.2f} stars** on held-out products (compared to ±{r['baseline_test_metrics']['MAE']:.2f} for naive guessing).
    """)

    tabs = st.tabs(["📊 Model Tournament", "🎯 Predictions & Residuals", "🔍 Feature Importance", "📋 Model Card & Export"])
    
    with tabs[0]:
        st.subheader("Grouped 5-Fold Cross-Validation Leaderboard")
        comp = pd.read_csv(path / "model_comparison.csv")
        st.dataframe(comp, hide_index=True, use_container_width=True)
        
        # Interactive bar chart of models
        fig_cv = px.bar(comp, x="model", y="CV_MAE", error_y="CV_SD", color="model",
                        title="Model Comparison · Grouped Cross-Validation MAE (Lower is better)")
        fig_cv.update_layout(showlegend=False)
        st.plotly_chart(fig_cv, use_container_width=True)
        
        st.subheader("Category-Wise Performance Breakdown")
        cat_metrics = pd.read_csv(path / "category_metrics.csv")
        st.dataframe(cat_metrics, hide_index=True, use_container_width=True)

    with tabs[1]:
        p = pd.read_csv(path / "test_predictions.csv")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(p, x="rating", y="prediction", color="category", hover_data=["asin"],
                             range_x=[1, 5], range_y=[1, 5],
                             title="Actual vs. Predicted Rating (Diagonal = Perfect Prediction)")
            fig.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(dash="dash", color="#49d9c2", width=2))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig_res = px.histogram(p, x="residual", nbins=20, title="Residual Errors (Actual - Predicted Rating)",
                                  color_discrete_sequence=["#49d9c2"])
            fig_res.add_vline(x=0, line_dash="dash", line_color="#e74c3c")
            st.plotly_chart(fig_res, use_container_width=True)

    with tabs[2]:
        imp = pd.read_csv(path / "feature_importance.csv")
        fig_imp = px.bar(imp.sort_values("MAE_increase"), x="MAE_increase", y="feature", orientation="h",
                         error_x="SD", title="Permutation Feature Importance · Impact on Test MAE",
                         color_discrete_sequence=["#49d9c2"])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Shows how much prediction error increases when each feature is shuffled. Demonstrates what drives product ratings.")

    with tabs[3]:
        st.markdown((path / "model_card.md").read_text(encoding="utf-8"))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for item in path.iterdir():
                if item.is_file():
                    z.write(item, item.name)
        st.download_button("📦 Download Experiment Bundle (.zip)", buf.getvalue(), f"{path.name}.zip", "application/zip")

# ==========================================
# PAGE: OVERVIEW
# ==========================================
if page == "Overview":
    header("AMAZON INDIA · AI INTELLIGENCE", "RateWise: Product Rating Intelligence", 
           "An end-to-end Machine Learning research system that predicts customer satisfaction ratings (1–5 stars) from pricing, discounts, brand, and engagement attributes.")
    
    st.markdown("""
    <div class="hero">
        <div class="demo-pill">✨ LIVE DEMONSTRATION READY</div>
        <h2>From Amazon Observations to Calibrated Rating Predictions</h2>
        <p>RateWise implements rigorous data provenance, zero-leakage grouped splitting by parent product families, multi-model tournament benchmarking, and calibrated 90% uncertainty intervals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    a, b, c, d = st.columns(4)
    a.metric("Total Observations", len(df))
    b.metric("Active Categories", df.category.nunique() if len(df) else 0)
    b_runs = get_runs()
    c.metric("Trained Experiments", len(b_runs))
    avg_r = pd.to_numeric(df.rating, errors="coerce").mean() if len(df) and "rating" in df else 0
    d.metric("Average Amazon Rating", f"{avg_r:.2f} ★" if avg_r else "N/A")

    st.markdown("### Research & Engineering Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">STAGE 01</div>
            <div class="step-title">Provenance & Data Integrity</div>
            <div class="step-desc">Captures live product attributes from Amazon India with SHA-256 cryptographic evidence hashing. Distinguishes rating counts from written review counts.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">STAGE 02</div>
            <div class="step-title">Grouped Multi-Model Tournament</div>
            <div class="step-desc">Guarantees zero data leakage by grouping product variants by Parent ASIN. Benchmarks Ridge, Random Forest, Extra Trees, and Gradient Boosting against a mean baseline.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">STAGE 03</div>
            <div class="step-title">Inference & Calibrated Intervals</div>
            <div class="step-desc">Estimates 1–5 star ratings for new or existing products with a 90% conformal uncertainty interval and out-of-distribution domain warnings.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Quick Navigation")
    col_a, col_b = st.columns(2)
    with col_a:
        st.success("👉 **Want to test a live prediction?**\nSwitch to **Predict ratings** to click 1-touch presets for OnePlus, boAt, Prestige, and Puma products!")
    with col_b:
        st.warning("👉 **Want to inspect model leaderboards?**\nSwitch to **Train & evaluate** to see why our model beats the baseline and which features matter most!")

# ==========================================
# PAGE: COLLECT DATA
# ==========================================
elif page == "Collect data":
    header("01 / DIRECT-SOURCE COLLECTION", "Record Genuine Amazon Observations", 
           "Record visible public product facts from Amazon India. Retain source URLs, timestamps, and cryptographic page evidence.")
    
    st.info("💡 **Demo Shortcut:** Click '⚡ Fill Sample Amazon Product' below to immediately test the collection pipeline with a verified Amazon product during your demo!")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("⚡ Fill Sample Amazon Product"):
            st.session_state["capture_title"] = "boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbuds with 42H Playtime"
            st.session_state["capture_brand"] = "boAt"
            st.session_state["capture_category"] = "Electronics"
            st.session_state["capture_parent_asin"] = "B09N3ZNPAR"
            st.session_state["capture_price"] = "1299"
            st.session_state["capture_list_price"] = "4490"
            st.session_state["capture_discount_pct"] = "71"
            st.session_state["capture_rating"] = "3.9"
            st.session_state["capture_rating_count"] = "184200"
            st.session_state["capture_review_count"] = "19200"
            st.session_state["capture_source"] = "https://www.amazon.in/dp/B09N3ZN98B"
            st.session_state["capture_transcript"] = (
                "ASIN: B09N3ZN98B\n"
                "Title: boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbuds\n"
                "Brand: boAt\n"
                "Price: INR 1,299\n"
                "MRP: INR 4,490\n"
                "Rating: 3.9 out of 5 stars (184,200 ratings, 19,200 reviews)"
            )
            st.rerun()

    method = st.radio("Capture method", ["Manual observation transcript", "Saved Amazon HTML upload"], horizontal=True)
    source = st.text_input("Amazon product URL", value=st.session_state.get("capture_source", "https://www.amazon.in/dp/B09N3ZN98B"), 
                           placeholder="https://www.amazon.in/dp/XXXXXXXXXX")
    
    raw = b""
    if method == "Saved Amazon HTML upload":
        uploaded = st.file_uploader("Saved product page (.html)", type=["html", "htm"])
        if uploaded:
            raw = uploaded.getvalue()
        if st.button("Extract fields from HTML", disabled=not bool(raw and source)):
            try:
                record, notes = parse_saved_html(raw, source)
                for key, val in record.items():
                    st.session_state[f"capture_{key}"] = str(val)
                for note in notes:
                    st.info(note)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.caption("Enter the visible product facts below. RateWise verifies fields, checks currency, and computes evidence SHA-256.")

    with st.form("capture"):
        values = {}
        a, b = st.columns(2)
        with a:
            values["title"] = st.text_input("Product title", value=st.session_state.get("capture_title", ""))
            values["brand"] = st.text_input("Brand", value=st.session_state.get("capture_brand", ""))
            values["category"] = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(st.session_state.get("capture_category", "Electronics")) if st.session_state.get("capture_category", "Electronics") in CATEGORIES else 0)
            values["parent_asin"] = st.text_input("Parent ASIN (if product has size/color variants)", value=st.session_state.get("capture_parent_asin", ""))
        with b:
            values["price"] = st.text_input("Current price (INR)", value=st.session_state.get("capture_price", ""))
            values["list_price"] = st.text_input("MRP / List price (INR, optional)", value=st.session_state.get("capture_list_price", ""))
            values["discount_pct"] = st.text_input("Discount % (optional)", value=st.session_state.get("capture_discount_pct", ""))
            values["rating"] = st.text_input("Observed rating (1.0 to 5.0)", value=st.session_state.get("capture_rating", ""))
            values["rating_count"] = st.text_input("Total rating count", value=st.session_state.get("capture_rating_count", ""))
            values["review_count"] = st.text_input("Written review count (optional)", value=st.session_state.get("capture_review_count", ""))

        captured_at = st.text_input("Observation timestamp (UTC ISO format)", value=datetime.now(timezone.utc).isoformat())
        transcript = st.text_area("Observation evidence / raw page text", value=st.session_state.get("capture_transcript", ""), height=100)
        confirmed = st.checkbox("I verified these product attributes against Amazon India.", value=True)
        submit = st.form_submit_button("Save Verified Product Observation", type="primary")

    if submit:
        try:
            if not confirmed:
                raise ValueError("Please check the confirmation checkbox before saving.")
            if method == "Saved Amazon HTML upload" and not raw:
                raise ValueError("Upload the saved HTML file first.")
            evidence = raw if method == "Saved Amazon HTML upload" else transcript.encode("utf-8")
            if not evidence.strip():
                raise ValueError("Please provide an observation transcript or upload HTML.")
            
            row = save_observation(
                {**values, "source_url": source, "currency": "INR", "collected_at": captured_at,
                 "collection_method": "saved_amazon_html" if method == "Saved Amazon HTML upload" else "manual_amazon"},
                evidence, DATA, extension="html" if raw else "txt"
            )
            st.success(f"✅ Successfully saved product {row['asin']}! Evidence SHA-256: {row['evidence_sha256'][:16]}…")
            df = load_observations(DATA)
        except (ValueError, sqlite3.IntegrityError) as exc:
            st.error(str(exc))

    st.divider()
    csv_download(pd.DataFrame(columns=COLUMNS), "Download Empty Collection Template CSV", "collection_template.csv")

# ==========================================
# PAGE: EXPLORE DATASET
# ==========================================
elif page == "Explore dataset":
    header("02 / DATA QUALITY & DISTRIBUTION", "Explore Amazon India Observations", 
           "Inspect completeness, category balance, rating distribution, and price relationships.")
    
    uploaded = st.file_uploader("Optional: Inspect an uploaded CSV dataset", type="csv")
    try:
        source_df = pd.read_csv(uploaded) if uploaded else df
        if source_df.empty:
            st.info("The dataset is empty. Use 'Collect data' or reload demo data in the sidebar.")
        else:
            clean, quality = validate_frame(source_df)
            
            # High-level KPIs
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Unique Products", quality["unique_products"])
            m2.metric("Product Families", clean.asin.nunique() - quality["parent_asin_missing"] if quality["parent_asin_missing"] < clean.asin.nunique() else clean.asin.nunique())
            m3.metric("Categories", clean.category.nunique())
            m4.metric("Mean Rating", f"{clean.rating.mean():.2f} ★")
            m5.metric("Median Price", f"₹{clean.price.median():,.0f}")

            st.markdown("### Interactive Visual Explorations")
            v1, v2 = st.columns(2)
            with v1:
                fig_hist = px.histogram(clean, x="rating", nbins=25, title="Observed Rating Distribution (1–5 Stars)",
                                       color_discrete_sequence=["#49d9c2"], marginal="box")
                fig_hist.add_vline(x=clean.rating.mean(), line_dash="dash", line_color="#f1c40f", 
                                   annotation_text=f"Mean: {clean.rating.mean():.2f}")
                st.plotly_chart(fig_hist, use_container_width=True)
            with v2:
                counts = clean.category.value_counts().rename_axis("category").reset_index(name="products")
                fig_cat = px.bar(counts, x="category", y="products", title="Category Coverage & Balance",
                                 color="category")
                fig_cat.update_layout(showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)

            v3, v4 = st.columns(2)
            with v3:
                fig_scatter = px.scatter(clean, x="price", y="rating", color="category", log_x=True,
                                        hover_data=["asin", "brand", "title"], title="Current Price (Log Scale) vs. Observed Rating")
                st.plotly_chart(fig_scatter, use_container_width=True)
            with v4:
                clean_disc = clean.dropna(subset=["discount_pct"])
                fig_disc = px.scatter(clean_disc, x="discount_pct", y="rating", color="category",
                                      hover_data=["brand", "price"], title="Discount % vs. Customer Rating")
                st.plotly_chart(fig_disc, use_container_width=True)

            st.markdown("### Data Quality & Provenance Audit")
            with st.expander("🔍 Click to inspect Data Quality Audit & Missingness"):
                st.json(quality)

            st.dataframe(clean, hide_index=True, use_container_width=True)
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                csv_download(clean, "Download Cleaned Dataset (CSV)", "amazon_products_clean.csv")
            with c_d2:
                if not uploaded:
                    bundle = io.BytesIO()
                    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("amazon_products.csv", safe_csv(clean))
                        for digest in clean.evidence_sha256.unique():
                            for ext in ["txt", "html"]:
                                ev = DATA / "evidence" / f"{digest}.{ext}"
                                if ev.exists():
                                    archive.write(ev, f"evidence/{ev.name}")
                    st.download_button("📦 Download Verified Dataset with Evidence Bundle (.zip)", bundle.getvalue(), "amazon_collection.zip", "application/zip")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(str(exc))

# ==========================================
# PAGE: TRAIN & EVALUATE
# ==========================================
elif page == "Train & evaluate":
    header("03 / REGRESSION TOURNAMENT", "Train Models with Grouped Cross-Validation", 
           "Evaluate candidate algorithms (Ridge, Random Forest, Extra Trees, Gradient Boosting) against a Mean Baseline with zero variant leakage.")
    
    st.markdown("""
    > **Data Splitting Guarantee:** RateWise partitions data strictly by product families (60% Train, 20% Interval Calibration, 20% Held-Out Evaluation). Product variants never cross partition boundaries.
    """)

    mode = st.selectbox("Experiment Mode", ["existing", "cold_start"], 
                        format_func=lambda x: "Existing products · includes engagement counts (ratings & reviews)" if x == "existing" else "Cold-start ablation · excludes review counts (new product simulation)")
    
    t_col1, t_col2 = st.columns([1, 3])
    with t_col1:
        train_btn = st.button("🚀 Train & Evaluate Models", type="primary", disabled=df.empty)
    
    if train_btn:
        try:
            run_dir = RUNS / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + mode)
            with st.spinner("Executing Grouped 5-Fold Cross-Validation, Fitting Candidates, and Calibrating 90% Intervals..."):
                train(df, run_dir, mode=mode, evidence_dir=DATA / "evidence")
            st.success("✅ Training & evaluation complete! Inspect the winning model below.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Experiment Results & Model Diagnostics")
    selected = run_picker("evaluation_run")
    if selected:
        show_results(selected)

# ==========================================
# PAGE: PREDICT RATINGS
# ==========================================
elif page == "Predict ratings":
    header("04 / PREDICTION STUDIO", "Live Amazon Product Rating Estimator", 
           "Estimate expected customer ratings (1–5 stars) with calibrated 90% uncertainty intervals and interactive sensitivity analysis.")
    
    selected = run_picker("prediction_run")
    if selected:
        artifact = joblib.load(selected / "model.joblib")
        if artifact["report"]["status"].startswith("SYNTHETIC"):
            st.error("This is a synthetic software-test artifact; please select a genuine model.")
            st.stop()
        
        st.markdown(f'<div class="demo-pill">● ACTIVE MODEL: {artifact["report"]["winner"]} ({artifact["mode"].upper()}) · CALIBRATED 90% INTERVAL RADIUS: ±{artifact["radius"]:.2f} STARS</div>', unsafe_allow_html=True)

        one, batch = st.tabs(["🎯 Single Product Live Predictor", "📁 Batch Prediction CSV"])
        
        with one:
            st.markdown("#### ✨ One-Touch Demo Presets")
            st.caption("Click any preset below to instantly load real-world Amazon India products:")
            
            p_cols = st.columns(6)
            presets = [
                ("boAt Rockerz 450", "Electronics", "boAt", 1499.0, 3990.0, 62.0, 128450, 15420),
                ("Samsung Galaxy M34", "Electronics", "Samsung", 16999.0, 24499.0, 31.0, 38900, 3900),
                ("Prestige Cooker", "Home & Kitchen", "Prestige", 2150.0, 2740.0, 22.0, 34200, 4100),
                ("Minimalist Serum", "Beauty & Personal Care", "Minimalist", 599.0, 599.0, 0.0, 42100, 5100),
                ("Puma Flyer Runner", "Sports & Fitness", "Puma", 2499.0, 3999.0, 38.0, 28900, 3100),
                ("Parker Vector Pen", "Office Products", "Parker", 340.0, 400.0, 15.0, 24500, 2800),
            ]
            
            for idx, (p_name, p_cat, p_brand, p_pr, p_lp, p_disc, p_rc, p_rev) in enumerate(presets):
                with p_cols[idx]:
                    if st.button(p_name, key=f"preset_{idx}", use_container_width=True):
                        st.session_state["pred_price"] = p_pr
                        st.session_state["pred_list_price"] = str(p_lp)
                        st.session_state["pred_discount_pct"] = str(p_disc)
                        st.session_state["pred_category"] = p_cat
                        st.session_state["pred_brand"] = p_brand
                        st.session_state["pred_rating_count"] = str(p_rc)
                        st.session_state["pred_review_count"] = str(p_rev)
                        st.rerun()

            with st.form("prediction_form"):
                a, b = st.columns(2)
                with a:
                    price = st.number_input("Current Price (INR)", min_value=1.0, 
                                            value=float(st.session_state.get("pred_price", 1499.0)), step=50.0)
                    listing = st.text_input("List price / MRP (INR, optional)", 
                                            value=str(st.session_state.get("pred_list_price", "3990.0")))
                    discount = st.text_input("Discount % (optional)", 
                                             value=str(st.session_state.get("pred_discount_pct", "62.0")))
                    cat_list = artifact["categories"]["category"]
                    curr_cat = st.session_state.get("pred_category", cat_list[0])
                    cat_idx = cat_list.index(curr_cat) if curr_cat in cat_list else 0
                    category = st.selectbox("Category", cat_list, index=cat_idx)
                with b:
                    brand = st.text_input("Brand", value=str(st.session_state.get("pred_brand", "boAt")))
                    count = st.text_input("Rating count (optional)", value=str(st.session_state.get("pred_rating_count", "128450")), 
                                         disabled=artifact["mode"] == "cold_start")
                    reviews = st.text_input("Written-review count (optional)", value=str(st.session_state.get("pred_review_count", "15420")), 
                                           disabled=artifact["mode"] == "cold_start")
                
                go = st.form_submit_button("🔮 Predict Customer Rating", type="primary", use_container_width=True)

            if go or "pred_price" in st.session_state:
                try:
                    inputs = pd.DataFrame([dict(
                        price=price, list_price=listing, discount_pct=discount, category=category,
                        brand=brand, rating_count=count, review_count=reviews, currency=artifact["report"]["currency"]
                    )])
                    result = predict(artifact, inputs).iloc[0]
                    
                    # Visual Rating Output Display
                    pred_val = float(result.prediction)
                    lower_val = float(result.lower_90)
                    upper_val = float(result.upper_90)
                    stars = render_star_display(pred_val)
                    
                    tier = "⭐ Highly Rated Product" if pred_val >= 4.2 else ("👍 Good Quality" if pred_val >= 3.8 else "⚠️ Moderate Rating")
                    tier_badge = "badge-green" if pred_val >= 4.2 else ("badge-blue" if pred_val >= 3.8 else "badge-amber")
                    
                    st.markdown(f"""
                    <div class="rating-card">
                        <div class="eyebrow">ESTIMATED CUSTOMER RATING</div>
                        <div class="rating-num">{pred_val:.2f} <span style="font-size:24px;color:#8ba2be;">/ 5.0</span></div>
                        <div class="rating-stars">{stars}</div>
                        <div style="margin: 10px 0;">
                            <span class="badge {tier_badge}">{tier}</span>
                        </div>
                        <div style="font-size: 15px; color: #b7c8da; margin-top: 14px;">
                            Nominal 90% Confidence Interval: <strong>{lower_val:.2f} ★</strong> to <strong>{upper_val:.2f} ★</strong> (Margin: ±{artifact['radius']:.2f} stars)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result.warnings:
                        st.warning(f"⚠️ **Domain Note:** {result.warnings}")

                    # Interactive What-If Simulator
                    st.markdown("---")
                    st.subheader("🔮 What-If Sensitivity Simulator")
                    st.write("Demonstrate how adjustments in **Discount %** impact the predicted customer rating:")
                    
                    sim_discounts = np.linspace(0, 70, 15)
                    sim_rows = []
                    for d in sim_discounts:
                        sim_rows.append(dict(
                            price=max(1.0, float(price) * (1.0 - d / 100.0)),
                            list_price=float(price),
                            discount_pct=d,
                            category=category,
                            brand=brand,
                            rating_count=count,
                            review_count=reviews,
                            currency=artifact["report"]["currency"]
                        ))
                    sim_df = pd.DataFrame(sim_rows)
                    sim_preds = predict(artifact, sim_df)
                    sim_df["Predicted_Rating"] = sim_preds["prediction"]
                    
                    fig_sim = px.line(sim_df, x="discount_pct", y="Predicted_Rating", 
                                      title=f"What-If: Predicted Rating vs. Discount % for {brand} ({category})",
                                      markers=True, labels={"discount_pct": "Discount % Offered", "Predicted_Rating": "Estimated Star Rating"})
                    fig_sim.update_traces(line_color="#49d9c2", marker=dict(size=8, color="#ffffff"))
                    st.plotly_chart(fig_sim, use_container_width=True)

                except ValueError as exc:
                    st.error(str(exc))

        with batch:
            st.markdown("#### Upload CSV for Bulk Rating Predictions")
            schema = pd.DataFrame(columns=["price", "list_price", "discount_pct", "category", "brand", "rating_count", "review_count", "currency"])
            csv_download(schema, "Download Batch CSV Template", "prediction_template.csv")
            upload = st.file_uploader("Products CSV to predict", type="csv", key="batch_csv")
            if upload and st.button("Run Batch Inference", type="primary"):
                try:
                    batch_df = pd.read_csv(upload)
                    output = predict(artifact, batch_df)
                    combined = pd.concat([batch_df.reset_index(drop=True), output], axis=1)
                    st.dataframe(combined, hide_index=True, use_container_width=True)
                    csv_download(combined, "Download Inferred Predictions", "predictions_output.csv")
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(str(exc))

# ==========================================
# PAGE: PROJECT GUIDE
# ==========================================
else:
    header("PROJECT & METHODOLOGY", "Complete Academic Report & Documentation", 
           "Full research methodology, mathematical formulation, baseline justification, and ethical considerations.")
    st.markdown((ROOT / "docs" / "PROJECT_REPORT.md").read_text(encoding="utf-8"))
