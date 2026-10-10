# AI-Based Amazon Product Rating Prediction

**Product:** RateWise — Amazon Product Rating Intelligence  
**Marketplace:** Amazon India  
**Status:** Software implementation supplied. Genuine Amazon collection, final model training and research findings are pending. No final accuracy claim is made.

## Abstract

This project implements a supervised regression workflow to estimate an Amazon product's aggregate star rating from its current price, reference price, discount, category, brand and available rating/review counts. Observations are collected directly from Amazon product pages, with source URLs, timestamps and retained evidence. A reproducible pipeline compares a mean baseline, regularized linear regression and tree-based ensembles. Product-family grouping reduces leakage between related variants. Separate calibration and test partitions support uncertainty estimation and independent evaluation. The application provides collection, exploratory analysis, training, diagnostics, single-product prediction and batch export. The empirical effectiveness of the method remains to be established using collected Amazon data.

## Problem statement and objectives

Develop a regression-based ML model to predict Amazon product ratings using price, discount, category, review count and other product attributes. Use direct Amazon observations rather than Kaggle or similar dataset repositories. The response variable is the displayed aggregate rating on a 1–5 scale. It is neither a sentiment class nor a fake-review label.

The central question is whether these attributes offer useful predictive information beyond a constant mean-rating baseline. A professional outcome includes reproducibility and an honest evaluation even if the more complex models do not outperform that baseline.

## Proposed product and users

RateWise is a locally run browser dashboard for students, evaluators and analysts. The workflow is Collect → Explore → Train & evaluate → Predict. Users can capture observed product data, inspect coverage and missingness, compare regression models, inspect held-out errors and estimate ratings for products whose attributes are known. It does not publish reviews, change Amazon ratings or claim affiliation with Amazon.

## Scope and research design

The primary experiment estimates the current rating of existing products. Rating count and written-review count may be included if observed at the same collection time. These variables reflect product maturity and are associated with ratings, so this is an observational estimation study rather than a causal claim.

A second, cold-start feature-ablation experiment excludes both count fields. Since its training observations still come from already-rated products, it must not be described as validated forecasting for genuinely new products. A true future-rating model would require time-separated snapshots: attributes at time t and rating at a later time, with chronological evaluation. That extension is outside the implemented snapshot experiment.

## Data collection

Plan for approximately 1,000 distinct Amazon India products: around 200 each in Electronics, Home & Kitchen, Beauty & Personal Care, Sports & Fitness and Office Products. This is a proposed study design, not a statistically guaranteed sample size. Begin with a pilot of 30–50 products to check field availability and collection effort, then expand. Training is gated at 100 distinct ASINs and 50 product families; this technical minimum is not a claim of research sufficiency.

Use a written collection plan, predefined search terms, a consistent collection window and a recorded inclusion rule. Avoid choosing products because their ratings look convenient. Search-result position, popularity and availability introduce selection bias. Record exclusions and missing attributes. Do not claim the sample represents all Amazon products.

Capture directly from product pages through manual observation or locally saved HTML. Amazon layouts and access controls may prevent extraction. The importer does not request pages, solve challenges, bypass sign-in or use proxies. Check applicable Amazon requirements and the institution's collection guidance before expanding collection. Official API availability does not by itself establish permission to retain data or use it for ML training; check the applicable agreement for the intended use.

Each saved observation retains a canonical source URL, collection timestamp, collection method and SHA-256 evidence reference. Manual evidence is a transcript of visible product facts; saved-HTML evidence is the uploaded file. Hash verification detects changed files but does not prove that their contents came from Amazon. A human source audit remains necessary. Evidence can contain account details in saved pages, so use public product pages without personal information and inspect files before sharing.

Amazon often displays a count of ratings. That is not necessarily the count of written reviews. Store these separately and leave unavailable counts blank. Never substitute an estimated value, an abbreviated count or an invented zero. If written-review counts cannot be obtained for enough products, disclose the missingness and obtain the supervisor's agreement on using rating count as an additional attribute; do not silently claim the review-count requirement has been met empirically.

## Data dictionary

| Field | Meaning | Model role |
|---|---|---|
| asin | Product identifier from Amazon URL | Audit / deduplication only |
| parent_asin | Known family/parent identifier, when available | Group splitting only |
| title | Displayed product title | Audit only; excluded from modeling |
| brand | Observed brand, otherwise Unknown | Categorical predictor |
| category | Consistent broad product category | Categorical predictor |
| price | Current displayed price in INR | Numeric predictor |
| list_price | Displayed MRP/reference price, if available | Numeric predictor |
| discount_pct | 100 × (1 − price/list_price), or observed discount when list price is unavailable | Numeric predictor |
| rating_count | Exact number of ratings, when visible | Existing-product predictor |
| review_count | Exact count of written reviews, when visible | Existing-product predictor |
| rating | Observed aggregate 1–5 rating | Target only |
| currency | INR for Amazon India | Validation only |
| source_url | Official product page URL | Provenance only |
| collected_at | Observation timestamp in UTC | Snapshot selection / audit |
| collection_method | manual_amazon or saved_amazon_html | Provenance only |
| evidence_sha256 | Hash of retained transcript/HTML | Evidence integrity only |

## Preprocessing

Validate product URLs, identity, numeric bounds, price relationships, count types, timestamps and evidence references. Remove older observations of the same ASIN, retaining the latest snapshot. Keep known parent/child product relations in the same group. Audit near-duplicate titles manually because missing family identifiers remain a leakage risk.

Apply log(1+x) to prices and counts to reduce skew. Numeric median imputation, missingness indicators, scaling and category encoding are fitted inside each cross-validation pipeline. Entirely missing numeric columns are retained by the imputer for consistent shapes. Unknown brands/categories use the encoder's supported unknown-category handling. Do not use the target rating, star-percentage breakdown, review text, ASIN or URL as predictors. Prices and discount are related mathematically, which should temper interpretations of importance.

## Model training and selection

Randomly partition product families with a fixed seed into approximately 60% training, 20% calibration and 20% test families. Proportions of rows may differ when families contain different numbers of variants. Compare a mean baseline, Ridge regression, Random Forest, Extra Trees and Gradient Boosting. Tune their provided hyperparameter grids using five-fold GroupKFold cross-validation on the training partition only, with MAE as the selection score. Save every grid candidate and the chosen parameters.

The lowest cross-validation MAE wins; the mean baseline is eligible to win. Predictions are bounded to 1–5 consistently during selection and inference. The selected model stays fitted on the training partition, preserving calibration and test independence. Refit on all observations would require fresh calibration and evaluation data and is deliberately not performed.

## Evaluation and uncertainty

Report held-out MAE, RMSE and R², alongside the mean-baseline test performance. MAE describes typical absolute error in rating points; RMSE penalizes larger errors; R² may be negative and is undefined for constant targets. Inspect observed-vs-predicted plots, residual distributions and category-specific errors. Small category slices should not support strong conclusions.

Construct nominal 90% split-conformal intervals from the maximum absolute residual within each calibration family, using the finite-sample rank ceil((n+1)×0.9). If that rank exceeds the number of families, use the full 1–5 outcome range. Report both product coverage and all-members-covered family coverage on the test partition, as well as interval width. The intended coverage interpretation relies on exchangeability of families; distribution shift and biased sampling can break it. It is not a per-product 90% certainty statement.

Permutation feature importance reports how much held-out MAE increases when a feature is shuffled. This describes model reliance rather than causality. Test-based inspection is descriptive; using it for iterative tuning would consume the test set and require a fresh holdout.

## Architecture and implementation

Streamlit provides the local interface and Plotly provides interactive charts. SQLite stores observations; an evidence directory retains original local evidence. pandas and NumPy handle data validation and transformation. scikit-learn pipelines implement preprocessing, regression, grouped selection, metrics and permutation importance. joblib stores trusted local models. A command-line interface supports validation, training and batch prediction.

Each experiment saves a model, metrics JSON, candidate search results, selected-model comparison, test predictions, category metrics, feature importance, split manifest and model card. A dataset hash, library versions, seed, feature list and split counts support reproduction. No uploaded executable model artifacts are accepted in the UI.

## Results and discussion — pending genuine collection

No Amazon dataset, final trained Amazon model or performance result is included in this initial delivery. The environment could not retrieve the attempted Amazon pages; the browser also failed to load them. Synthetic records are confined to software tests and must not appear in the research dataset, results chapter or claimed model performance.

After genuine collection, train both experiment modes and insert their actual outputs here. Discuss whether each model beats the baseline, the magnitude of its errors, interval coverage, category imbalance, count missingness and any failure to generalize. Do not promise a specific accuracy or R². Negative or modest results can still answer the research question.

## Limitations and future work

Public product metadata may explain only a small fraction of rating variation. Ratings can be concentrated in a narrow range; small absolute errors can coexist with weak R². Review counts, brand, price and popularity can be confounded. Selected categories, missing variants, rounded displayed ratings, unavailable reviews and collection dates affect interpretation. Cold-start and out-of-range predictions need extra caution.

Future work could add genuinely time-separated collection, a fresh-marketplace external test set, a family-audited larger sample and learning curves to assess whether more data improves performance. Any additional predictors must be available at the intended prediction time and must not reveal the target.

## Requirements traceability

| Requirement | Implemented component | Completion evidence |
|---|---|---|
| Direct official Amazon source | Collection form, saved-page parser, source URL and evidence storage | Software present; real dataset pending |
| Regression-based model | Ridge and ensemble regressors, mean baseline | Training and evaluation code present |
| Price and discount | Validated numeric fields and derived discount | Covered by software tests |
| Category | Consistent category field and encoder | Covered by software tests |
| Review count | Separate optional written-review count | Real availability must be established during collection |
| Other attributes | Brand, list price and rating count | Included in feature pipeline |
| Rating prediction | Single and batch prediction with 1–5 bounds | Software tested using labelled fixtures |
| Professional evaluation | Grouped CV, holdout metrics, diagnostics and model card | Generates after genuine training |
| Advanced functionality | Calibration intervals, provenance, experiments, warnings, exports | Implemented; scientific validity requires real evaluation |

## References

- [scikit-learn pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Streamlit API documentation](https://docs.streamlit.io/develop/api-reference)
- [Amazon Creators API onboarding](https://affiliate-program.amazon.com/creatorsapi/docs/en-us/onboarding) — possible official integration route subject to eligibility and permitted use; not implemented or assumed suitable for ML.
- [Amazon India Conditions of Use](https://www.amazon.in/gp/help/customer/display.html?nodeId=GLSBYFE9MGKKQXXM) — review directly before collection; this page was not retrievable in the build environment.
- [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)
