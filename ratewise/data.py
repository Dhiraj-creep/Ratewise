from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

MARKETS = {"amazon.in": "INR", "amazon.com": "USD"}
COLUMNS = ["asin", "parent_asin", "title", "brand", "category", "price", "list_price",
           "discount_pct", "rating", "rating_count", "review_count", "currency",
           "source_url", "collected_at", "collection_method", "evidence_sha256"]
NUMERIC = ["price", "list_price", "discount_pct", "rating_count", "review_count"]
FEATURES = NUMERIC + ["brand", "category"]

def number(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).strip().replace(",", "").replace("₹", "").replace("$", "").replace("%", "").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        raise ValueError(f"Expected an exact number, received {value!r}. Do not estimate abbreviated counts.")
    return float(text)

def amazon_identity(url):
    p = urlparse(str(url))
    host = (p.hostname or "").lower().removeprefix("www.")
    match = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:[/?]|$)", p.path, re.I)
    if p.scheme != "https" or host not in MARKETS or not match or p.username or p.password or p.port:
        raise ValueError("Use an HTTPS amazon.in or amazon.com product URL containing /dp/ASIN.")
    asin = match.group(1).upper()
    return host, asin, f"https://www.{host}/dp/{asin}"

def normalize_record(record, require_target=True):
    row = {k: record.get(k, "") for k in COLUMNS}
    for key in COLUMNS:
        if not isinstance(row[key], str) and pd.isna(row[key]):
            row[key] = ""
    market, asin, url = amazon_identity(row["source_url"])
    if row["asin"] and str(row["asin"]).upper() != asin:
        raise ValueError("ASIN does not match the source URL.")
    row.update(asin=asin, source_url=url)
    if row["currency"] != MARKETS[market]:
        raise ValueError("Currency does not match the Amazon marketplace.")
    for key in ["title", "category"]:
        row[key] = str(row[key]).strip()
        if not row[key]:
            raise ValueError(f"{key} is required.")
    row["brand"] = str(row["brand"]).strip() or "Unknown"
    row["parent_asin"] = str(row["parent_asin"]).upper().strip()
    if row["parent_asin"] and not re.fullmatch(r"[A-Z0-9]{10}", row["parent_asin"]):
        raise ValueError("Parent ASIN must contain 10 letters/digits, or be blank.")
    for key in NUMERIC + ["rating"]:
        row[key] = number(row[key])
    if not np.isfinite(row["price"]) or row["price"] <= 0:
        raise ValueError("Current price must be positive.")
    for key in ["rating_count", "review_count"]:
        val = row[key]
        if not np.isnan(val) and (not np.isfinite(val) or val < 0 or val != int(val)):
            raise ValueError(f"{key} must be a nonnegative whole number or blank.")
    if np.isfinite(row["list_price"]):
        if row["list_price"] < row["price"]:
            raise ValueError("List price must not be less than current price.")
        calculated = 100 * (1 - row["price"] / row["list_price"])
        if np.isfinite(row["discount_pct"]) and abs(calculated - row["discount_pct"]) > 2:
            raise ValueError("Discount differs from the price-derived discount by more than 2 percentage points.")
        row["discount_pct"] = round(calculated, 4)
    if np.isfinite(row["discount_pct"]) and not 0 <= row["discount_pct"] <= 100:
        raise ValueError("Discount must be between 0 and 100.")
    if require_target and (not np.isfinite(row["rating"]) or not 1 <= row["rating"] <= 5):
        raise ValueError("An observed rating between 1 and 5 is required for training.")
    if row["collection_method"] not in {"manual_amazon", "saved_amazon_html"}:
        raise ValueError("Collection method must be manual_amazon or saved_amazon_html.")
    if not re.fullmatch(r"[a-f0-9]{64}", str(row["evidence_sha256"])):
        raise ValueError("Evidence SHA-256 is required. Capture through the collection screen first.")
    stamp = pd.to_datetime(row["collected_at"], utc=True, errors="coerce")
    if pd.isna(stamp) or stamp.to_pydatetime() > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("Collection time must be a valid timestamp, not in the future.")
    row["collected_at"] = stamp.isoformat()
    return row

def validate_frame(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("The dataset is empty. Collect Amazon observations first.")
    rows, issues = [], []
    for idx, row in frame.iterrows():
        try:
            rows.append(normalize_record(row.to_dict()))
        except (ValueError, TypeError) as exc:
            issues.append({"row": str(idx), "issue": str(exc)})
    if issues:
        raise ValueError("Invalid rows (first 10): " + str(issues[:10]))
    result = pd.DataFrame(rows)
    if result["currency"].nunique() != 1:
        raise ValueError("Train separate experiments for each marketplace/currency.")
    before = len(result)
    result = result.sort_values("collected_at", kind="stable").drop_duplicates("asin", keep="last").reset_index(drop=True)
    return result, {"input_rows": before, "unique_products": len(result),
                    "older_snapshots_removed": before - len(result),
                    "missing_values": result[FEATURES].isna().sum().to_dict(),
                    "parent_asin_missing": int(result.parent_asin.eq("").sum())}

def features(frame, mode="existing"):
    keys = FEATURES if mode == "existing" else [x for x in FEATURES if x not in {"rating_count", "review_count"}]
    out = frame.reindex(columns=keys).copy()
    for key in set(NUMERIC) & set(keys):
        out[key] = out[key].map(number)
        if (out[key].dropna() < 0).any() or np.isinf(out[key]).any():
            raise ValueError(f"{key} must contain finite nonnegative values.")
    if out.price.isna().any() or (out.price <= 0).any():
        raise ValueError("A positive current price is required for every prediction.")
    if (out.list_price.notna() & (out.list_price < out.price)).any():
        raise ValueError("List price cannot be below current price.")
    known = out.list_price.notna()
    calculated = 100 * (1 - out.price / out.list_price)
    if (known & out.discount_pct.notna() & ((out.discount_pct - calculated).abs() > 2)).any():
        raise ValueError("Discount is inconsistent with the entered prices.")
    out.loc[known, "discount_pct"] = calculated[known]
    if (out.discount_pct.dropna() > 100).any():
        raise ValueError("Discount cannot exceed 100 percent.")
    for key in ["rating_count", "review_count"]:
        if key in out and ((out[key].dropna() % 1) != 0).any():
            raise ValueError(f"{key} must be a whole number.")
    for key in ["brand", "category"]:
        out[key] = out[key].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    for key in ["price", "list_price", "rating_count", "review_count"]:
        if key in out:
            out[key] = np.log1p(out[key])
    return out

def groups(frame):
    # Union all explicit parent-child relations, even if one row omits its parent.
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]
    for row in frame.itertuples():
        a = root(row.asin)
        if row.parent_asin:
            parent[a] = root(row.parent_asin)
    return frame.asin.map(root)

def save_observation(record, evidence: bytes, root: Path, extension="txt"):
    if not evidence.strip():
        raise ValueError("Save page evidence or an observation transcript before collecting.")
    if len(evidence) > 20 * 1024 * 1024:
        raise ValueError("Evidence exceeds 20 MB.")
    digest = hashlib.sha256(evidence).hexdigest()
    row = normalize_record({**record, "evidence_sha256": digest})
    folder = root / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    ext = "html" if extension == "html" else "txt"
    (folder / f"{digest}.{ext}").write_bytes(evidence)
    with sqlite3.connect(root / "observations.sqlite") as con:
        con.execute("CREATE TABLE IF NOT EXISTS observations (" + ",".join(f'"{x}" TEXT' for x in COLUMNS) + ", UNIQUE(asin,collected_at))")
        vals = [None if isinstance(row[x], float) and math.isnan(row[x]) else row[x] for x in COLUMNS]
        con.execute("INSERT INTO observations (" + ",".join(COLUMNS) + ") VALUES (" + ",".join("?" for _ in COLUMNS) + ")", vals)
    return row

def load_observations(root):
    path = Path(root) / "observations.sqlite"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    with sqlite3.connect(path) as con:
        return pd.read_sql_query("SELECT * FROM observations", con)

def safe_csv(frame):
    out = frame.copy()
    for c in out.select_dtypes(include=["object", "string"]).columns:
        out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
    return out.to_csv(index=False).encode("utf-8")
