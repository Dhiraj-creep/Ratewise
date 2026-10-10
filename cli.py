import argparse
from pathlib import Path
import joblib
import pandas as pd
from ratewise.data import validate_frame, safe_csv
from ratewise.model import train, predict

def main():
    parser = argparse.ArgumentParser(description="RateWise reproducible experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("csv", type=Path)
    fit = sub.add_parser("train")
    fit.add_argument("csv", type=Path)
    fit.add_argument("--evidence-dir", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True, help="New experiment directory")
    fit.add_argument("--mode", choices=["existing", "cold_start"], default="existing")
    fit.add_argument("--seed", type=int, default=42)
    infer = sub.add_parser("predict")
    infer.add_argument("csv", type=Path)
    infer.add_argument("--model", type=Path, required=True, help="Trusted local model.joblib produced by this project")
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        frame = pd.read_csv(args.csv)
        if args.command == "validate":
            _, report = validate_frame(frame)
            print(report)
        elif args.command == "train":
            if args.output.exists():
                raise ValueError("Use a new output directory to preserve earlier experiments.")
            _, report = train(frame, args.output, mode=args.mode, seed=args.seed, evidence_dir=args.evidence_dir)
            print(report["test_metrics"])
            print("Beats mean baseline:", report["test_beats_mean_baseline"])
        else:
            artifact = joblib.load(args.model)
            if artifact["report"]["status"].startswith("SYNTHETIC"):
                raise ValueError("Synthetic test artifacts cannot be used for Amazon predictions.")
            result = pd.concat([frame.reset_index(drop=True), predict(artifact, frame)], axis=1)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(safe_csv(result))
            print(f"Saved {len(result)} predictions to {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

if __name__ == "__main__":
    main()
