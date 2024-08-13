import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv


class APIKeyManager:
    def __init__(self):
        # 環境変数から暗号化キーを取得、または新しいキーを生成
        self.encryption_key = os.environ.get("MOSAICAI_ENCRYPTION_KEY") or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        # APIキーを保存する辞書
        self.api_keys = {}
        # .envファイルを読み込む
        load_dotenv()

    def set_api_key(self, model: str, api_key: str):
        # APIキーを暗号化して辞書に保存
        encrypted_key = self.fernet.encrypt(api_key.encode())
        self.api_keys[model] = encrypted_key

    def get_api_key(self, model: str) -> str:
        # モデルに対応するAPIキーが存在しない場合はNoneを返す
        if model not in self.api_keys:
            return None
        # 暗号化されたAPIキーを復号して返す
        encrypted_key = self.api_keys[model]
        return self.fernet.decrypt(encrypted_key).decode()

    def load_from_env(self):
        # .envファイルの環境変数を参考にしてAPIキーを読み込む
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY"
        }
        for model, env_key in env_keys.items():
            if api_key := os.environ.get(env_key):
                self.set_api_key(model, api_key)
