"""Fixtures below are invented software-test observations, never a research dataset."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from ratewise.data import normalize_record, validate_frame, groups, features, safe_csv, save_observation, load_observations
from ratewise.collector import parse_saved_html
from ratewise.model import train, predict, split_indices, verify_evidence

def record(i=0):
    return dict(asin=f"T{i:09d}", parent_asin="", title=f"SYNTHETIC TEST ONLY {i}",
                brand="Fixture", category="Test category", price=100, list_price=125,
                discount_pct=20, rating=4.1, rating_count=100, review_count="",
                currency="INR", source_url=f"https://www.amazon.in/dp/T{i:09d}",
                collected_at="2026-09-01T12:00:00Z", collection_method="manual_amazon",
                evidence_sha256=hashlib.sha256(b"SYNTHETIC TEST ONLY").hexdigest())

def fixture_frame(n=150):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        r = record(i)
        r.update(price=round(float(rng.uniform(100, 2000)),2), rating=round(float(rng.uniform(2,5)),1),
                 list_price="", discount_pct=float(rng.uniform(0,60)), rating_count=int(rng.integers(0,5000)),
                 category=f"Test category {i % 5}", brand=f"Test brand {i % 8}")
        rows.append(r)
    return pd.DataFrame(rows)

def test_currency_and_counts_cleaning():
    r = record()
    r.update(price="₹1,000", list_price="₹1,250", review_count="1,200")
    result = normalize_record(r)
    assert result["price"] == 1000 and result["review_count"] == 1200 and result["discount_pct"] == 20

@pytest.mark.parametrize("field,value", [("rating",6), ("rating_count",-1), ("review_count",1.5),
                                            ("price",0), ("discount_pct",99), ("currency","USD"),
                                            ("source_url","https://amazon.in.fake.org/dp/T000000000"),
                                            ("collected_at","2999-01-01"), ("evidence_sha256","fake"),
                                            ("asin","B000000000"), ("review_count","1.2K")])
def test_invalid_observation_rejected(field, value):
    r = record()
    r[field] = value
    with pytest.raises(ValueError):
        normalize_record(r)

def test_snapshot_deduplication():
    a,b = record(), record()
    b.update(collected_at="2026-09-02T12:00:00Z", rating=4.2)
    frame, report = validate_frame(pd.DataFrame([a,b]))
    assert len(frame) == 1 and frame.rating.iloc[0] == 4.2
    assert report["older_snapshots_removed"] == 1

def test_parent_families_do_not_cross_splits():
    frame = fixture_frame()
    for i in range(0, 100, 2):
        frame.loc[i + 1, "parent_asin"] = frame.loc[i, "asin"]
    a,b,c,g = split_indices(frame, 42)
    assert not set(g.iloc[a]) & set(g.iloc[b])
    assert not set(g.iloc[a]) & set(g.iloc[c])
    assert not set(g.iloc[b]) & set(g.iloc[c])
    assert g.iloc[0] == g.iloc[1]

def test_target_and_identifiers_never_enter_features():
    frame = fixture_frame()
    X = features(frame)
    assert not {"rating", "asin", "title", "source_url", "evidence_sha256"} & set(X)
    assert not {"review_count", "rating_count"} & set(features(frame,"cold_start"))

def test_html_parser_keeps_rating_count_separate():
    content = b'''<input id="ASIN" value="T000000000"><span id="productTitle">Test fixture only</span>
    <span id="acrPopover">4.2 out of 5 stars</span><span id="acrCustomerReviewText">1,234 ratings</span>
    <div id="corePriceDisplay_desktop_feature_div"><span class="a-price"><span class="a-offscreen">100</span></span></div>'''
    r,_ = parse_saved_html(content, record()["source_url"])
    assert r["rating_count"] == "1234" and r["review_count"] == "" and r["rating"] == "4.2"
    with pytest.raises(ValueError):
        parse_saved_html(content.replace(b"T000000000", b"B000000000"), record()["source_url"])

def test_access_challenge_rejected():
    with pytest.raises(ValueError, match="challenge"):
        parse_saved_html(b'<input id="captchacharacters">', record()["source_url"])

def test_evidence_round_trip_and_tamper(tmp_path):
    save_observation(record(), b"SYNTHETIC TEST ONLY", tmp_path)
    frame,_ = validate_frame(load_observations(tmp_path))
    verify_evidence(frame, tmp_path / "evidence")
    path = next((tmp_path / "evidence").glob("*.txt"))
    path.write_text("changed")
    with pytest.raises(ValueError, match="absent or have changed"):
        verify_evidence(frame, tmp_path / "evidence")

def test_csv_formula_escaping():
    assert "'=1+1" in safe_csv(pd.DataFrame({"title":["=1+1"]})).decode()

def test_model_end_to_end(tmp_path):
    frame = fixture_frame()
    artifact, report = train(frame, tmp_path / "test_run", fixture=True, fast=True)
    result = predict(artifact, frame.iloc[:5])
    assert len(result) == 5
    assert result.prediction.between(1,5).all()
    assert (result.lower_90 <= result.prediction).all() and (result.upper_90 >= result.prediction).all()
    assert report["status"].startswith("SYNTHETIC")
    assert sum(report["split_counts"].values()) == 150
    changed = frame.iloc[:1].copy()
    changed["brand"] = "Unseen fixture brand"
    assert "Unseen brand" in predict(artifact, changed).warnings.iloc[0]
    changed["currency"] = "USD"
    with pytest.raises(ValueError, match="currency"):
        predict(artifact, changed)

def test_production_training_requires_evidence(tmp_path):
    with pytest.raises(ValueError, match="evidence directory"):
        train(fixture_frame(), tmp_path)

def test_minimum_dataset_gate(tmp_path):
    with pytest.raises(ValueError, match="100 unique"):
        train(fixture_frame(10), tmp_path, fixture=True)
