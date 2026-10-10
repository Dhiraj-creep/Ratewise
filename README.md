# RateWise

**AI-Based Amazon Product Rating Prediction — Amazon India**

RateWise is a local ML research dashboard that estimates 1–5 star ratings from price, discount, category, brand and available rating/review counts.

**Delivery status:** application, training pipeline, collection workflow, tests and methodology are implemented. Genuine Amazon observations, final research model and empirical results are not included. Direct page retrieval failed in the build environment. No Kaggle or third-party dataset was used. Synthetic fixtures exist only inside tests.

## Start the app

Install Python 3.12. On Windows, double-click `start.bat` in this directory. First launch installs dependencies into a local `.venv`; internet access is needed for that installation. The app opens at http://localhost:8501.

Manual setup, from this directory:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, use `python3 -m venv .venv`, then `.venv/bin/python` in place of `.venv\Scripts\python.exe`. Keep the app local. It is a single-user academic tool, without public-deployment authentication.

## What the product contains

- Overview: research workspace, architecture pipeline, and system metrics.
- Collect data: direct Amazon observations or saved HTML, 1-click sample product filler, timestamps, source validation and SHA-256 evidence storage.
- Explore dataset: missingness audit, category balance, rating distribution, price/discount scatter plots, and export with evidence.
- Train & evaluate: mean baseline, Ridge, Random Forest, Extra Trees and Gradient Boosting; grouped model tournament and held-out diagnostics.
- Predict ratings: one-touch demo presets (boAt, OnePlus, Prestige, etc.), visual star rating cards, nominal 90% confidence intervals, and interactive What-If sensitivity simulator.
- Project guide: comprehensive academic report draft, methodology, and limitations.

## Collect real data

Read `docs/COLLECTION_PROTOCOL.md` first. Open Amazon India in your normal browser. Copy exactly observed product facts into Collect data, or save a product page as HTML and import it. Saved-page extraction is best effort, not guaranteed against every Amazon layout. Verify all extracted values before saving.

Use around 1,000 products across five categories as a collection target. The application requires at least 100 unique ASINs and 50 families to begin training. That minimum permits a technical run; it does not guarantee a scientifically sufficient dataset.

**Rating count and written-review count differ.** Missing written-review counts remain blank. Their availability must be established during the pilot before claiming the review-count requirement is satisfied in the final study.

This version does not automatically scrape Amazon and does not integrate an Amazon API. It accepts directly observed facts and locally saved page evidence. Hashes verify file integrity, not authenticity; audit a sample against the original pages.

## Train and export

Use Train & evaluate after collecting and auditing data. Run existing-product and cold-start experiments separately. The latter omits counts but does not establish prospective performance on new products.

For a downloaded collection bundle, extract it and use:

```powershell
.venv\Scripts\python.exe cli.py validate amazon_products.csv
.venv\Scripts\python.exe cli.py train amazon_products.csv --evidence-dir evidence --output runs\existing_01 --mode existing
.venv\Scripts\python.exe cli.py train amazon_products.csv --evidence-dir evidence --output runs\cold_start_01 --mode cold_start
.venv\Scripts\python.exe cli.py predict products_to_predict.csv --model runs\existing_01\model.joblib --output predictions.csv
```

Prediction input columns: `price,list_price,discount_pct,category,brand,rating_count,review_count,currency`. Current price and matching currency are required. Optional values can be blank. Fill category and brand when known. Never load another person's untrusted joblib/pickle file; the UI only loads local experiment outputs.

Model files are tied to the versions used to create them. `requirements-tested.txt` records this delivery's tested dependency versions; retrain if library versions change. Model selection uses training-only cross-validation; do not select a run by repeatedly inspecting its test score.

## Experiment outputs

Each run saves `model.joblib`, `metrics.json`, `model_card.md`, `model_comparison.csv`, `cv_search.csv`, `test_predictions.csv`, `category_metrics.csv`, `feature_importance.csv` and `split_manifest.csv`. The metrics record the dataset hash, seed, versions, features and partition sizes. Interpret the selected model against the baseline, including when it fails to improve on it.

## Test

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Tests use clearly labelled synthetic fixtures, separate from `data/` and `runs/`. They verify software behavior and cannot support claims about Amazon prediction quality. See `docs/VERIFICATION.md` for checks performed on this delivery.

## Files

```text
app.py                      Streamlit interface
cli.py                      reproducible command-line workflow
ratewise/data.py            schema, validation, evidence, SQLite, feature preparation
ratewise/collector.py       saved-Amazon-HTML parser
ratewise/model.py           selection, calibration, evaluation and inference
tests/                      synthetic software verification only
docs/PROJECT_REPORT.md      academic report draft with pending-results section
docs/COLLECTION_PROTOCOL.md collection and audit procedure
docs/VIVA_AND_DEMO.md       explanation and presentation outline
docs/VERIFICATION.md        build verification and remaining work
data/                       observations and retained evidence created at runtime
runs/                       genuine experiments created at runtime
```

## Finish the academic submission

1. Collect a pilot directly from Amazon India and check review-count availability.
2. Expand and audit the dataset; record collection choices and exclusions.
3. Run both experiments, inspect actual metrics and document limitations.
4. Update the report's pending results with generated outputs and actual collection dates.
5. Add student names, institution details and required formatting; rehearse the demonstration.

The software is a working foundation for the requested project. The research submission is complete only after genuine collection and evaluation. Do not submit invented results or label software-test data as Amazon data.
