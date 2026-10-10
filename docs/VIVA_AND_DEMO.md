# Viva preparation and demonstration

## Five-minute demonstration

1. State the problem: predicting a numeric Amazon rating from product attributes.
2. Show a genuine source page and its corresponding observation/evidence. Explain rating count versus written reviews.
3. Show dataset size, missingness, category distribution and rating distribution.
4. Show the existing-product model comparison, baseline and held-out metrics.
5. Predict one product without entering its known rating. Explain the interval and any warnings.
6. Show the cold-start experiment and explain why it is an ablation, not a validated forecast of future success.
7. Open the model card and explain one limitation honestly.

Until genuine data is collected, demonstrate collection and empty states only; do not imply that a trained Amazon model exists.

## Presentation outline

1. Title, team and institution.
2. Problem statement and regression objective.
3. Product: RateWise workflow and intended users.
4. Direct Amazon data collection with provenance and evidence.
5. Dataset profile — insert actual collection statistics.
6. Features, review-count distinction and missing values.
7. Pipeline and grouped train/calibration/test separation.
8. Models and cross-validation selection.
9. Results — insert generated metrics, including baseline comparison.
10. Prediction and uncertainty demonstration.
11. Limitations and future time-separated study.
12. Conclusion based on actual results, references and questions.

## Questions you should be able to answer

**Why regression?** The target is a numerical aggregate star rating, so the model estimates a continuous value rather than a sentiment label.

**Why not use accuracy?** Classification accuracy is not the appropriate primary metric here. MAE and RMSE measure error in rating points; R² compares squared-error performance with a constant prediction on the evaluated data.

**What is the baseline?** A model that predicts the training mean rating for every product. Complex models must demonstrate value beyond it.

**Why is the rating not a feature?** It is the target. Including it, star-distribution percentages or equivalent information would leak the answer.

**Why exclude review text?** Existing review text can directly describe the target satisfaction signal and may not be available at the intended prediction time. The project focuses on structured attributes.

**Why split by product family?** Variants can share reviews and many attributes. Splitting them across training and test can exaggerate performance. Missing family identifiers remain a limitation.

**Why three partitions?** Training selects/fits the model; calibration sets the interval width; the final test measures performance. The last two do not choose the model.

**Why log-transform counts and prices?** Their distributions are commonly skewed; log(1+x) reduces extreme scale differences while allowing zero counts.

**What does the 90% interval mean?** It is calibrated using held-out family residuals with a nominal coverage target under an exchangeability assumption. It is not a promise that this specific product has a 90% chance of lying in its particular interval.

**Does more discount cause a higher rating?** This model cannot establish causality. Price, brand, quality and popularity may be confounded.

**What if R² is negative?** The evaluated predictions have larger squared error than the test-target mean benchmark. Report this; do not hide it. Ratings may have a narrow range and weak relationships with available features.

**Can it predict new-product success?** The count-free ablation uses already-rated products. Validating new-product or future-rating prediction requires genuinely prospective or time-separated data.

**How do you know the data came from Amazon?** Source URLs, timestamps and retained evidence support a human audit. Hashes prove evidence integrity, not source authenticity by themselves.

**What makes the work advanced?** Reproducible grouped model selection, separate uncertainty calibration, honest baseline evaluation, provenance checks, error diagnostics and prediction-range warnings. Complexity is justified only if the actual evaluation supports it.
