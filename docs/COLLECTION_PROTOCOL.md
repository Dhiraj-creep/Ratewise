# Direct Amazon India collection protocol

## 1. Pilot and scope

Start with 30–50 products across the proposed categories. Determine whether current price, reference price, category, aggregate rating, rating count and written-review count are actually visible. Clarify unavailable written-review counts with the supervisor before full collection. Missing data is not zero. Do not substitute rating count without disclosing the substitution.

The suggested full study is 1,000 distinct products, about 200 per broad category. Use Electronics, Home & Kitchen, Beauty & Personal Care, Sports & Fitness and Office Products unless the supervisor chooses a different scope. This quota balances category counts but does not make the sample representative of Amazon.

Record a collection log: collector, observation window, predefined search terms, category, search-result page/position, inclusion decision and reason for exclusion. Use a consistent sampling rule such as every third eligible non-sponsored result within a predefined result-page range. Document how the starting position is selected. Do not select only popular or highly rated products.

## 2. Open and observe

Open a product page directly on amazon.in. Check the selected variant and displayed currency. Record the /dp/ ASIN, title, broad category, brand, current price, MRP/reference price, aggregate rating and the exact labels attached to displayed counts. Keep a consistent delivery context and note price variability in the study log without storing personal addresses.

Do not infer an unavailable brand from a seller name. Leave missing reference prices/counts blank. Do not estimate abbreviated counts such as 1.2K. Avoid coupon-adjusted assumptions: use the visible current price before optional coupons and document this choice. Products without an observed target rating cannot train a supervised rating model; log them as excluded.

If parent-family identity is available, record it. Prefer one variant from each family; otherwise keep known related variants together. Review near-duplicate titles and shared rating counts for unidentified families. Repeated snapshots of the same ASIN are deduplicated to the latest observation by the pipeline.

## 3. Capture evidence

Manual route: paste a transcript of the visible facts into the collection screen and record the observation time. Include the count labels so rating count and written-review count can be audited separately.

Saved-page route: save the accessible product page as HTML in your normal browser, upload it to Collect data, and supply its exact source URL. Click Extract fields, correct incomplete extraction and verify the entries. Use the time when the page was saved rather than the later import time. The parser will reject recognized access challenges and pages without a product title. It does not execute the saved HTML.

The app retains the evidence with its SHA-256 filename and appends the observation to SQLite. An evidence file proves what was retained, not that the collector transcribed it correctly. Do not place personal account or delivery details in evidence. Collection must follow applicable site requirements and institutional instructions; no circumvention workflow is included.

## 4. Audit and export

Have another team member check at least 10% of observations against their retained evidence and, when feasible, the source page. This is a practical QA recommendation, not a guarantee of validity. Check identity, current price, MRP, discount, rating, count labels, category consistency and collection time. Correct mistakes by entering a new observation with a later observation timestamp after rechecking the page; preserve the original evidence for audit.

Use Explore dataset to check category coverage, rating distribution, duplicate removal and missingness. Download the dataset-with-evidence bundle, retain the original bundle, and train from a fixed audited snapshot. Never share a model without its metrics and dataset provenance.

## 5. Record the final dataset facts

Fill these in only after collection:

- Start and end observation dates: pending.
- Number of raw observations and distinct products: pending.
- Number of product families and missing parent IDs: pending.
- Actual category counts: pending.
- Written-review-count and rating-count availability: pending.
- Exclusion counts and reasons: pending.
- Source-audit sample size and corrections: pending.
- Final dataset hash and experiment seed: generated on training.

If access is blocked, stop that collection attempt and log it. Use accessible pages or a collection route explicitly available to you. Do not manufacture missing rows to reach a target size.
