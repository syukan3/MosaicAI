# API Reference

## MosaicAI

### `__init__(model: str, config: Dict[str, Any] = None)`

MosaicAIクライアントを初期化します。

### `generate_text(prompt: str) -> str`

指定されたモデルを使用してテキストを生成します。

### `generate_image_description(image_path: str, prompt: str) -> str`

指定されたモデルを使用して画像の説明を生成します。

### `generate_json(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]`

指定されたモデルを使用してJSONオブジェクトを生成します。

### `generate_with_image(prompt: str, image_path: str) -> str`

指定されたモデルを使用して画像付きのテキストを生成します。

### `generate_with_image_json(prompt: str, image_path: str, schema: Dict[str, Any]) -> Dict[str, Any]`

指定されたモデルを使用して画像付きのJSONを生成します。

### `set_api_key(model: str, api_key: str)`

指定されたモデルのAPI keyを設定します。

### `classmethod from_config_file(config_path: str) -> 'MosaicAI'`

設定ファイルからMosaicAIクライアントを初期化します。

### `_parse_json_response(response: str) -> dict`

文字列形式のJSON応答をパースします。

### `_generate_schema_description(schema: Dict[str, Union[str, Dict]]) -> str`

出力スキーマの説明を生成します。

### `_convert_types(data: Dict[str, Any], schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]`

データの型をスキーマに従って変換します。

### `_convert_value(value: Any, type_str: str) -> Any`

単一の値を指定された型に変換します。
