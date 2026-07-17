"""
Configuration utility for the Loan Intelligence Platform.

Loads project configuration from YAML files.
"""

from pathlib import Path
import yaml


# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Configuration directory
CONFIG_DIR = ROOT_DIR / "configs"




def load_config(filename: str = "config.yaml"):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    filename : str
        Name of the configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    config_path = CONFIG_DIR / filename

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


config = load_config()

config["data"]["raw_path"] = (ROOT_DIR / config["data"]["raw_path"]).resolve()
config["data"]["interim_path"] = (ROOT_DIR / config["data"]["interim_path"]).resolve()
config["data"]["processed_path"] = (ROOT_DIR / config["data"]["processed_path"]).resolve()

config["models"]["save_dir"] = (ROOT_DIR / config["models"]["save_dir"]).resolve()

config["artifacts"]["path"] = (ROOT_DIR / config["artifacts"]["path"]).resolve()

config["reports"]["figures_dir"] = (ROOT_DIR / config["reports"]["figures_dir"]).resolve()

config["logging"]["log_dir"] = (ROOT_DIR / config["logging"]["log_dir"]).resolve()