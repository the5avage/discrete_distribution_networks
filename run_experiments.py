# run_experiments.py

import argparse
import glob
import json
import os
import subprocess
import sys

import yaml
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_glob", type=str, default="configs/*.yaml",
                        help="Glob pattern for config files.")
    parser.add_argument("--train_script", type=str, default="train_ddn.py",
                        help="Training script to run.")
    parser.add_argument("--results_csv", type=str, default="experiments_summary.csv",
                        help="Where to write the summary CSV.")
    args = parser.parse_args()

    config_paths = sorted(glob.glob(args.configs_glob))
    print(f"Found {len(config_paths)} config files.")

    rows = []
    for cfg_path in config_paths:
        print(f"\n=== Running experiment for config: {cfg_path} ===")
        ret = subprocess.run(
            [sys.executable, args.train_script, "--config", cfg_path],
            check=False
        )
        if ret.returncode != 0:
            print(f"Training failed for {cfg_path} (exit code {ret.returncode}).")
            continue

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        exp_name = cfg.get("experiment_name", os.path.splitext(os.path.basename(cfg_path))[0])
        output_dir = cfg.get("output_dir", os.path.join("experiments", exp_name))
        results_path = os.path.join(output_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"Results file {results_path} not found.")
            continue
        with open(results_path, "r") as f:
            results = json.load(f)

        row = {
            "config": cfg_path,
            "experiment_name": exp_name,
            "output_dir": output_dir,
            "best_val_loss": results.get("best_val_loss", None),
            "test_loss": results.get("test_loss", None),
        }
        rows.append(row)

    # write CSV
    if rows:
        fieldnames = ["config", "experiment_name", "output_dir", "best_val_loss", "test_loss"]
        with open(args.results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nWrote summary to {args.results_csv}")
    else:
        print("\nNo successful experiments to summarize.")


if __name__ == "__main__":
    main()