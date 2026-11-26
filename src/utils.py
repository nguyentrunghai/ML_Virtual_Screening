from pathlib import Path
import yaml


def load_config(config_path: str | Path, verify: bool = True):
    """
    Load model paths from a YAML config, resolve them to Path objects,
    and optionally verify that every file and directory exists on disk.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root_dir = Path(config["root_dir"]).expanduser().resolve()
    if verify and not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    model_paths: dict[str, dict[str, Path]] = {}
    missing: list[str] = []

    for target, models in config["model_paths"].items():
        model_paths[target] = {}
        for model_name, rel_path in models.items():
            full_path = (root_dir / rel_path).resolve()
            model_paths[target][model_name] = full_path

            if verify and not full_path.exists():
                missing.append(f"  - [{target} / {model_name}]: {full_path}")

    if missing:
        raise FileNotFoundError(
            "The following configured model paths do not exist on disk:\n"
            + "\n".join(missing)
        )

    return root_dir, model_paths