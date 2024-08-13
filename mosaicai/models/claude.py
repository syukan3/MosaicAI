import base64
import mimetypes
import json
import os
import logging
from typing import Dict, Any, Union
from anthropic import Anthropic
from .base import AIModelBase
from ..utils.api_key_manager import APIKeyManager


class Claude(AIModelBase):
    def __init__(self, api_key_manager: APIKeyManager, model: str = "claude-3-5-sonnet-20240620"):
        """
        Claudeモデルの初期化
        :param api_key_manager: API鍵を管理するAPIKeyManagerインスタンス
        :param model: 使用するモデルの名前（デフォルトは"claude-3-5-sonnet-20240620"）
        """
        self.api_key_manager = api_key_manager
        api_key = self.api_key_manager.get_api_key("claude")
        if not api_key:
            raise ValueError("Claude APIキーが設定されていません。")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_image_size = 20 * 1024 * 1024  # 20MB

    def get_model(self) -> str:
        """
        self.modelの値を返す
        :return: 使用するモデルの名前
        """
        return self.model

    def generate(self, message: str) -> str:
        """
        指定されたメッセージに対してClaudeの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :return: Claudeが生成した応答テキスト
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message}
                ],
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"テキスト生成中にエラーが発生しました: {str(e)}")
            raise

    def generate_with_image(self, message: str, image_path: str) -> str:
        """
        画像を含むメッセージに対してClaudeの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :return: Claudeが生成した応答テキスト
        """
        try:
            self._validate_image_file(image_path)
            mime_type = self._get_mime_type(image_path)
            base64_image = self._encode_image(image_path)

            logging.info(f"画像ファイルのパス: {image_path}")
            logging.info(f"MIMEタイプ: {mime_type}")
            logging.info(f"ファイルサイズ: {os.path.getsize(image_path)} bytes")

            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"画像を含むメッセージの生成中にエラーが発生しました: {str(e)}")
            raise

    def generate_json(self, message: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        指定されたメッセージに対してClaudeのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: Claudeが生成したJSON応答（辞書形式）
        """
        schema_description = self._generate_schema_description(output_schema)
        system_message = f"応答は以下のJSON形式で生成してください: \n{schema_description}"
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=[
                    {"role": "user", "content": message}
                ],
                max_tokens=1000
            )
            json_response = self._parse_json_response(response.content[0].text)
            return self._convert_types(json_response, output_schema)
        except Exception as e:
            logging.error(f"JSON生成中にエラーが発生しました: {str(e)}")
            raise

    def generate_with_image_json(self, message: str, image_path: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        画像を含むメッセージに対してClaudeのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: Claudeが生成したJSON応答（辞書形式）
        """
        try:
            self._validate_image_file(image_path)
            schema_description = self._generate_schema_description(output_schema)
            system_message = f"応答は以下のJSON形式で生成してください: \n{schema_description}"
            mime_type = self._get_mime_type(image_path)
            base64_image = self._encode_image(image_path)

            logging.info(f"画像ファイルのパス: {image_path}")
            logging.info(f"MIMEタイプ: {mime_type}")
            logging.info(f"ファイルサイズ: {os.path.getsize(image_path)} bytes")

            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            json_response = self._parse_json_response(response.content[0].text)
            return self._convert_types(json_response, output_schema)
        except Exception as e:
            logging.error(f"画像を含むJSONメッセージの生成中にエラーが発生しました: {str(e)}")
            raise

    def _validate_image_file(self, image_path: str):
        """画像ファイルの妥当性を検証する"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"画像ファイルを読み込む権限がありません: {image_path}")
        if os.path.getsize(image_path) > self.max_image_size:
            raise ValueError(f"画像ファイルが大きすぎます。{self.max_image_size/1024/1024}MB以下にしてください。")

    def _get_mime_type(self, image_path: str) -> str:
        """画像ファイルのMIMEタイプを取得する"""
        mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')
                                                                ) else mimetypes.guess_type(image_path)[0]
        return mime_type if mime_type else 'application/octet-stream'

    def _encode_image(self, image_path: str) -> str:
        """画像ファイルをbase64エンコードする"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def _generate_schema_description(self, schema: Dict[str, Union[str, Dict]]) -> str:
        """スキーマの説明を生成する"""
        return json.dumps(schema, indent=2)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON形式の応答をパースする"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logging.error(f"JSONのパースに失敗しました: {response}")
            raise ValueError("生成された応答が有効なJSON形式ではありません。")

    def _convert_types(self, data: Dict[str, Any], schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """データの型をスキーマに従って変換する"""
        for key, value in data.items():
            if key in schema:
                expected_type = schema[key]
                if expected_type == "int":
                    data[key] = int(value)
                elif expected_type == "float":
                    data[key] = float(value)
                elif expected_type == "bool":
                    data[key] = bool(value)
                elif expected_type == "List[str]":
                    if not isinstance(value, list):
                        data[key] = [str(value)]
                    else:
                        data[key] = [str(item) for item in value]
        return data
