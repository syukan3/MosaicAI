import yaml
from typing import Dict, Any

# 設定ファイルを読み込む関数
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

# 設定をファイルに保存する関数
def save_config(config: Dict[str, Any], config_path: str):
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)

# デフォルトの設定
DEFAULT_CONFIG = {
    "default_model": "chatgpt",
    "logging": {
        "level": "INFO",
        "file": "mosaicai.log"
    },
    "request_timeout": 30,
    "max_retries": 3
}