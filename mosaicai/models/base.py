from abc import ABC, abstractmethod
import json
from typing import Dict, Any, Union, Type
from pydantic import BaseModel


class AIModelBase(ABC):
    @abstractmethod
    def generate(self, message: str) -> str:
        """
        メッセージを受け取り、AIモデルからの応答を生成する抽象メソッド

        :param message: ユーザーからの入力メッセージ
        :return: AIモデルが生成した応答テキスト
        """
        pass

    # @abstractmethod
    # def generate_with_image(self, message: str, image_path: str) -> str:
    #     """
    #     画像とメッセージを受け取り、AIモデルからの応答を生成する抽象メソッド

    #     :param message: ユーザーからの入力メッセージ
    #     :param image_path: 入力画像のファイルパス
    #     :return: AIモデルが生成した応答テキスト
    #     """
    #     pass

    def generate_json(self, message: str, output_schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        """
        メッセージを受け取り、指定されたスキーマに従ってJSON形式の応答を生成する抽象メソッド

        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: AIモデルが生成したJSON応答（辞書形式）
        """
        pass

    def _parse_json_response(self, response: str) -> dict:
        """
        文字列形式のJSON応答をパースする内部メソッド
        :param response: JSON形式の文字列
        :return: パースされたJSONオブジェクト（辞書形式）
        :raises ValueError: 応答が有効なJSONでない場合
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            error_message = f"生成された応答が有効なJSONではありません。エラー: {str(e)}\n応答内容: {response}"
            raise ValueError(error_message)

    def _generate_schema_description(self, schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> str:
        """
        出力スキーマの説明を生成する内部メソッド

        :param schema: 期待される出力のスキーマ（dictまたはPydanticモデル）
        :return: スキーマの説明文字列
        :raises TypeError: 入力またはPydanticモデルでない場合
        """
        if isinstance(schema, dict):
            return self._generate_schema_description_from_dict(schema)
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            return self._generate_schema_description_from_pydantic(schema)
        else:
            raise TypeError("入力は辞書またはPydanticモデルである必要があります")

    def _generate_schema_description_from_dict(self, schema: Dict[str, Union[str, Dict]]) -> str:
        """
        辞書からスキーマの説明を生成する内部メソッド

        :param schema: 期待される出力のスキーマ（辞書形式）
        :return: スキーマの説明文字列
        """
        description = "{\n"
        for key, value in schema.items():
            if isinstance(value, dict):
                description += f' "{key}": ' + "{\n"
                for sub_key, sub_value in value.items():
                    description += f'   "{sub_key}": "{sub_value}",\n'
                description = description.rstrip(',\n') + "\n },\n"
            else:
                description += f' "{key}": "{value}",\n'
        description = description.rstrip(',\n') + "\n}"
        return description

    def _generate_schema_description_from_pydantic(self, model: Type[BaseModel]) -> str:
        """
        Pydanticモデルからスキーマの説明を生成する内部メソッド

        :param model: Pydanticモデル
        :return: スキーマの説明文字列
        """
        schema = model.model_json_schema()
        return self._format_json_schema(schema)

    def _format_json_schema(self, schema: Dict[str, Any], indent: int = 0) -> str:
        """
        JSONスキーマを人間が読みやすい形式に変換する内部メソッド

        :param schema: JSONスキーマ
        :param indent: インデントレベル
        :return: 整形されたスキーマの説明文字列
        """
        description = "{\n"
        for key, value in schema.get("properties", {}).items():
            description += f'{" " * (indent + 2)}"{key}": '
            if value.get("type") == "object":
                description += self._format_json_schema(value, indent + 2)
            else:
                description += f'"{value.get("type", "any")}",\n'
        description += f'{" " * indent}}}'
        return description

    def _convert_types(self, data: Dict[str, Any], schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        """
        データの型をスキーマに従って変換する内部メソッド

        :param data: 変換するデータ
        :param schema: 期待される出力のスキーマ（辞書またはPydanticモデル）
        :return: 型変換されたデータ
        :raises ValueError: スキーマに適合しないデータの場合
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()['properties']
        elif isinstance(schema, dict):
            schema_dict = schema
        else:
            raise TypeError("スキーマは辞書またはPydanticモデルである必要があります")

        converted_data = {}
        for key, value in schema_dict.items():
            if key not in data:
                raise ValueError(f"キー '{key}' が応答に含まれていません。")
            if isinstance(value, dict) and 'type' in value:
                converted_data[key] = self._convert_value(data[key], value)
            elif isinstance(value, dict):
                converted_data[key] = self._convert_types(data[key], value)
            else:
                converted_data[key] = self._convert_value(data[key], value)
        return converted_data

    def _convert_value(self, value: Any, type_info: Union[str, Dict[str, Any]]) -> Any:
        """
        単一の値を指定された型に変換する内部メソッド

        :param value: 変換する値
        :param type_info: 変換先の型を示す文字列または辞書
        :return: 変換された値
        :raises ValueError: 変換できない場合
        """
        if isinstance(type_info, dict):
            type_str = type_info.get('type', 'any')
        else:
            type_str = type_info

        try:
            if type_str == "integer" or type_str == "int":
                return int(float(value))
            elif type_str == "number" or type_str == "float":
                return float(value)
            elif type_str == "boolean" or type_str == "bool":
                if isinstance(value, bool):
                    return value
                return str(value).lower() == "true"
            elif type_str == "string" or type_str == "str":
                return str(value)
            elif type_str == "array" or type_str == "list":
                if isinstance(type_info, dict):
                    item_type = type_info.get('items', {}).get('type', 'any')
                else:
                    item_type = 'any'
                return [self._convert_value(item, item_type) for item in value]
            else:
                return value
        except ValueError:
            raise ValueError(f"値 '{value}' を型 '{type_str}' に変換できません。")
