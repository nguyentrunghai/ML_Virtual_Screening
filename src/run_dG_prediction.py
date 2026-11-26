from ml_inference import predict_ensemble
from utils import load_config
from pathlib import Path
import pandas as pd
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dG Prediction for a drug target",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file containing root_dir and model_paths.",
    )
    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Drug target name.",
    )

    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Input csv file containing the SMILES of compounds to predict.",
    )

    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="Column name of the SMILES in the input csv file.",
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default="output.csv",
        help="Output csv file containing the predicted dG values.",
    )

    parser.add_argument(
        "--pred_col",
        type=str,
        default="ensemble_predicted_dG",
        help="Column name of ensemble prediction in the output csv file.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config
    target = args.target
    input_csv = args.input_csv
    smiles_col = args.smiles_col

    output_csv = args.output_csv
    pred_col = args.pred_col
    
    _, all_model_paths = load_config(config_path)
    if target not in all_model_paths:
        raise ValueError(f"Target {target} not found in the config file.")

    model_paths = all_model_paths[target]
    df = pd.read_csv(input_csv)

    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col} not found in the input csv file.")

    df = predict_ensemble(model_paths, df, smiles_col, pred_col)

    print(f"Saving result to {output_csv}")
    df.to_csv(output_csv, index=False)
    return None

if __name__ == "__main__":
    main()
