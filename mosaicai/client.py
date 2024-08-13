from typing import Dict, Any
import json
from .models import ChatGPT, Claude, Gemini, Perplexity, AIModelBase
from .utils.api_key_manager import APIKeyManager
from .exceptions import ModelNotSupportedError


class MosaicAI:
    """
    MosaicAIクラスは、複数のAIモデルを統合して管理するためのクラスです。
    異なるAIモデルを使用してテキスト生成、画像説明生成、JSON生成などの機能を提供します。
    """

    def __init__(self, model: str, config: Dict[str, Any] = None):
        """
        MosaicAIクラスのコンストラクタ。

        :param model: 使用するモデルの名前
        :param config: 設定情報を含む辞書（オプション）
        """
        self.config = config or {}
        self.api_key_manager = APIKeyManager()
        self.api_key_manager.load_from_env()
        self.models = {}
        self.initialize_model(model)

    def initialize_model(self, model: str) -> AIModelBase:
        if model not in self.models:
            if model.startswith("gpt-"):
                self.models[model] = ChatGPT(self.api_key_manager, model)
            elif model.startswith("claude-"):
                self.models[model] = Claude(self.api_key_manager, model)
            elif model.startswith("gemini-"):
                self.models[model] = Gemini(self.api_key_manager, model)
            elif model.startswith("llama-"):
                self.models[model] = Perplexity(self.api_key_manager, model)
            else:
                raise ValueError(f"サポートされていないモデル: {model}")
        return self.models[model]

    def _set_api_keys_from_config(self):
        """
        設定から各モデルのAPIキーを設定します。
        """
        for model, api_key in self.config.get('api_keys', {}).items():
            self.set_api_key(model, api_key)

    def get_model(self) -> str:
        """
        使用中のモデル名を返します。

        :return: 使用中のモデル名
        """
        return list(self.models.keys())[0]

    def generate_text(self, prompt: str) -> str:
        """
        指定されたモデルを使用してテキストを生成します。

        :param prompt: 生成のためのプロンプト
        :return: 生成されたテキスト
        :raises ModelNotSupportedError: 指定されたモデルがサポートされていない場合
        """
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")

        model = self.get_model()
        if model not in self.models:
            raise ModelNotSupportedError(f"Model '{model}' is not supported.")
        return self.models[model].generate(prompt)

    def generate_image_description(self, image_path: str, prompt: str) -> str:
        """
        指定されたモデルを使用して画像の説明を生成します。

        :param image_path: 画像ファイルのパス
        :param prompt: 生成のためのプロンプト
        :return: 生成された画像の説明
        :raises ModelNotSupportedError: 指定されたモデルがサポートされていない場合
        """
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")
        model = self.get_model()
        if model not in self.models:
            raise ModelNotSupportedError(f"Model '{model}' is not supported.")
        return self.models[model].generate_with_image(image_path, prompt)

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定されたモデルを使用してJSONを生成します。

        :param prompt: 生成のためのプロンプト
        :param schema: 生成するJSONのスキーマ
        :return: 生成されたJSON
        :raises ModelNotSupportedError: 指定されたモデルがサポートされていない場合
        """
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")
        model = self.get_model()
        if model not in self.models:
            raise ModelNotSupportedError(f"Model '{model}' is not supported.")
        return self.models[model].generate_json(prompt, schema)

    def generate_with_image(self, prompt: str, image_path: str) -> str:
        """
        指定されたモデルを使用して画像付きのテキストを生成します。

        :param prompt: 生成のためのプロンプト
        :param image_path: 画像ファイルのパス
        :return: 生成されたテキスト
        :raises ModelNotSupportedError: 指定されたモデルがサポートされていない場合
        """
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")
        model = self.get_model()
        if model not in self.models:
            raise ModelNotSupportedError(f"モデル '{model}' はサポートされていません。")

        if not hasattr(self.models[model], 'generate_with_image'):
            raise ModelNotSupportedError(f"モデル '{model}' は画像付きの生成をサポートしていません。")

        return self.models[model].generate_with_image(prompt, image_path)

    def generate_with_image_json(self, prompt: str, image_path: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定されたモデルを使用して画像付きのJSONを生成します。

        :param prompt: 生成のためのプロンプト
        :param image_path: 画像ファイルのパス
        :param schema: 生成するJSONのスキーマ
        :return: 生成されたJSON
        :raises ModelNotSupportedError: 指定されたモデルがサポートされていない場合
        """
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。有効なプロンプトを入力してください。")

        model = self.get_model()
        if model not in self.models:
            raise ModelNotSupportedError(f"モデル '{model}' はサポートされていません。")

        if not hasattr(self.models[model], 'generate_with_image_json'):
            raise ModelNotSupportedError(f"モデル '{model}' は画像付きのJSON生成をサポートしていません。")

        return self.models[model].generate_with_image_json(prompt, image_path, schema)

    def set_api_key(self, model: str, api_key: str):
        """
        指定されたモデルのAPIキーを設定します。

        :param model: APIキーを設定するモデルの名前
        :param api_key: 設定するAPIキー
        """
        self.api_key_manager.set_api_key(model, api_key)

    @classmethod
    def from_config_file(cls, config_path: str) -> 'MosaicAI':
        """
        設定ファイルからMosaicAIインスタンスを作成します。

        :param config_path: 設定ファイルのパス
        :return: 作成されたMosaicAIインスタンス
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)
