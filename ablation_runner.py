import yaml
import os
import subprocess
from datetime import datetime
import shutil

ABLATION_CONFIGS = [
    {
        "name": "all_rules",
        "rules": {}  # 默认全启用
    },
    {
        "name": "no_window",
        "rules": {"desk_window_weight": 0.0}
    },
    {
        "name": "no_path_check",
        "rules": {"path_clear_bonus": 0.0, "path_block_penalty": 0.0}
    },
    {
        "name": "no_spacing",
        "rules": {
            "inter_item_too_close_penalty": 0.0,
            "inter_item_close_penalty": 0.0
        }
    },
    {
        "name": "only_wall_and_spacing",
        "rules": {
            "path_clear_bonus": 0.0,
            "path_block_penalty": 0.0,
            "desk_window_weight": 0.0,
            "nightstand_near_bed_bonus": 0.0,
            "nightstand_far_penalty": 0.0,
            "wardrobe_near_penalty": 0.0,
            "wardrobe_far_bonus": 0.0
        }
    }
]

def write_yaml(config: dict, filename="reward_config.yaml"):
    with open(filename, "w") as f:
        yaml.dump(config, f)

def run_experiment(config_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n🚀 Running config: {config_name} [{timestamp}]")

    log_path = f"logs/{config_name}_{timestamp}.txt"
    config_backup = f"logs/{config_name}_{timestamp}_config.yaml"

    os.makedirs("logs", exist_ok=True)
    shutil.copy("reward_config.yaml", config_backup)

    with open(log_path, "w") as log_file:
        subprocess.run(["python", "train.py"], stdout=log_file, stderr=subprocess.STDOUT)

    print(f"✅ Log saved to {log_path}")
    print(f"🧾 Config saved to {config_backup}")

def main():
    for config in ABLATION_CONFIGS:
        name = config["name"]
        overrides = config["rules"]
        write_yaml(overrides)
        run_experiment(name)

if __name__ == "__main__":
    main()
