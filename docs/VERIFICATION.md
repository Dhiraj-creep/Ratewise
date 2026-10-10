# Delivery verification

Build date: 19 September 2026.

## Verified

- 23 automated checks passed: schema/range validation, currency and ASIN checks, exact counts, snapshot deduplication, product-family split separation, target exclusion, HTML extraction, access-challenge rejection, evidence integrity, CSV formula escaping, inference bounds, currency rejection, minimum dataset gates and six empty-state dashboard screens.
- Full five-model searches completed in both existing-product and cold-start modes on 180 explicitly synthetic software-test records, using 108 training, 36 calibration and 36 test records in each fixture run.
- Saved models reloaded and generated bounded predictions.
- Streamlit's AppTest rendered populated evaluation metrics, charts and model cards without script exceptions.
- The prediction screen refused a synthetic test artifact.

The synthetic experiments were kept outside the deliverable research data/runs directories. Their prediction-quality metrics are intentionally not presented as project findings.

## Environment observations

The packaged Python runtime encountered a Windows temporary-directory ACL issue during dependency installation and pytest setup. Build-only wrappers used inherited directory ACLs for temporary directories. Those wrappers are not part of the project application. Dependencies were installed in the task workspace, leaving the user's global Python installations unchanged.

The app's Streamlit configuration explicitly disables development mode to support execution from a nonstandard package path. AppTest is programmatic UI verification, not a browser screenshot review. Browser automation timed out, so a visual browser review was not completed.

## Still required for an academic completion claim

1. A real, auditable Amazon India dataset. Direct retrieval and browser access did not succeed during this build.
2. A pilot establishing written-review-count availability, followed by documented collection and source audit.
3. Training and evaluation on that frozen real dataset, including baseline comparison and interval coverage.
4. Real findings and screenshots in the report, plus any institution-specific formatting and team details.

No Kaggle data, fabricated Amazon records, trained Amazon research model or claimed empirical accuracy is bundled.
