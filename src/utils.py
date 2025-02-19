import json

def load_config(config_path):
    """
    JSON形式の設定ファイルを読み込み、辞書として返します。
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config