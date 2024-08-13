import base64
from typing import Dict, Any, Union
import json
from openai import OpenAI
from .base import AIModelBase
from ..utils.api_key_manager import APIKeyManager


class ChatGPT(AIModelBase):
    def __init__(self, api_key_manager: APIKeyManager, model: str = "gpt-4o"):
        """
        ChatGPTモデルの初期化
        :param api_key_manager: API鍵を管理するAPIKeyManagerインスタンス
        :param model: 使用するモデルの名前（デフォルトは"gpt-4o"）
        """
        self.api_key_manager = api_key_manager
        api_key = self.api_key_manager.get_api_key("openai")
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません。")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_model(self) -> str:
        """
        self.modelの値を返す
        :return: 使用するモデルの名前
        """
        return self.model

    def generate(self, message: str) -> str:
        """
        指定されたメッセージに対してChatGPTの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :return: ChatGPTが生成した応答テキスト
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content

    def generate_with_image(self, message: str, image_path: str) -> str:
        """
        画像を含むメッセージに対してChatGPTの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :return: ChatGPTが生成した応答テキスト
        """
        with open(image_path, "rb") as image_file:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}}
                        ],
                    }
                ]
            )
        return response.choices[0].message.content

    def generate_json(self, message: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        指定されたメッセージに対してChatGPTのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: ChatGPTが生成したJSON応答（辞書形式）
        """
        schema_description = self._generate_schema_description(output_schema)
        system_message = f"応答は以下のJSON形式で生成してください: \n{schema_description}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"}
        )

        return self._parse_json_response(response.choices[0].message.content, output_schema)

    def generate_with_image_json(self, message: str, image_path: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        画像を含むメッセージに対してChatGPTのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: ChatGPTが生成したJSON応答（辞書形式）
        """
        schema_description = self._generate_schema_description(output_schema)
        system_message = f"応答は以下のJSON形式で生成してください: \n{schema_description}"

        with open(image_path, "rb") as image_file:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {"type": "text", "text": message},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"
                        }}
                    ]}
                ],
                response_format={"type": "json_object"}
            )

        return self._parse_json_response(response.choices[0].message.content, output_schema)

    def _generate_schema_description(self, schema: Dict[str, Union[str, Dict]]) -> str:
        """
        出力スキーマの説明を生成する
        :param schema: 期待される出力のスキーマ
        :return: スキーマの説明文字列
        """
        description = "{\n"
        for key, value in schema.items():
            if isinstance(value, dict):
                description += f'  "{key}": ' + "{\n"
                for sub_key, sub_value in value.items():
                    description += f'    "{sub_key}": "{sub_value}",\n'
                description = description.rstrip(',\n') + "\n  },\n"
            else:
                description += f'  "{key}": "{value}",\n'
        description = description.rstrip(',\n') + "\n}"
        return description

    def _parse_json_response(self, response: str, schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        文字列形式のJSON応答をパースし、スキーマに従って型変換する
        :param response: JSON形式の文字列
        :param schema: 期待される出力のスキーマ
        :return: パースされたJSONオブジェクト（辞書形式）
        :raises ValueError: 応答が有効なJSONでない場合、またはスキーマに適合しない場合
        """
        try:
            parsed_response = json.loads(response)
            return self._convert_types(parsed_response, schema)
        except json.JSONDecodeError:
            raise ValueError("生成された応答が有効なJSONではありません。")

    def _convert_types(self, data: Dict[str, Any], schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        データの型をスキーマに従って変換する
        :param data: 変換するデータ
        :param schema: 期待される出力のスキーマ
        :return: 型変換されたデータ
        """
        converted_data = {}
        for key, value in schema.items():
            if key not in data:
                raise ValueError(f"キー '{key}' が応答に含まれていません。")
            if isinstance(value, dict):
                converted_data[key] = self._convert_types(data[key], value)
            else:
                converted_data[key] = self._convert_value(data[key], value)
        return converted_data

    def _convert_value(self, value: Any, type_str: str) -> Any:
        """
        単一の値を指定された型に変換する
        :param value: 変換する値
        :param type_str: 変換先の型を示す文字列
        :return: 変換された値
        :raises ValueError: 変換できない場合
        """
        try:
            if type_str == "int":
                return int(value)
            elif type_str == "float":
                return float(value)
            elif type_str == "bool":
                return bool(value)
            elif type_str == "str":
                return str(value)
            elif type_str.startswith("List["):
                item_type = type_str[5:-1]
                return [self._convert_value(item, item_type) for item in value]
            else:
                return value
        except ValueError:
            raise ValueError(f"値 '{value}' を型 '{type_str}' に変換できません。")