from ml_inference import predict
from utils import load_config
from pathlib import Path
import pandas as pd
import argparse
import sys

def inference_test(model_paths, test_data_dir, mse_cutoff=1e-8):
    test_results = {}
    for target in model_paths:
        print("*"*10)
        print(f'Drug Target: {target}')
        print()
        test_results[target] = {}
        for model_name in model_paths[target]:
            print("-"*10)
            print(f"Model {model_name}")
            model_path = model_paths[target][model_name]
            print(f"Loading model from {model_path}")
            test_data_path = Path(test_data_dir) / f"{target}/{model_name}/test_data.csv"
            print(f"Loading test data from {test_data_path}")
            dftest = pd.read_csv(test_data_path)
            print(f"Test size before inference: {dftest.shape[0]}")
            df00 = predict(model_path, model_name, dftest)
            print(f"Test size after inference: {df00.shape[0]}")
            if ((df00["predicted_dG"] - df00[f"{model_name}_pred"])**2).mean() < mse_cutoff:
                test_res = "PASS"
            else:
                test_res = "FAIL"
            print(test_res)
            print()

            test_results[target][model_name] = test_res
        print()
        print()
    return test_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ML inference verification tests across drug targets and models.",
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
        "--mse-cutoff",
        type=float,
        default=1e-8,
        help="Maximum allowed MSE between predicted_dG and reference predictions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir, model_paths = load_config(args.config)
    test_data_dir = Path(root_dir) / "data/test/"
    test_results = inference_test(model_paths, test_data_dir, args.mse_cutoff)
    
    print()
    print("#"*20)
    print("Test result summary:")
    for target in test_results:
        print(target)
        for model_name in test_results[target]:
            print(f"{model_name}: {test_results[target][model_name]}")
        print()
    any_failed = any(
        status != "PASS"
        for models in test_results.values()
        for status in models.values()
    )
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())