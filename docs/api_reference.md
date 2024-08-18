# API リファレンス

## MosaicAI

### `__init__(model: str, config: Dict[str, Any] = None)`

MosaicAIクライアントを初期化します。

### `generate_text(prompt: str) -> str`

指定されたモデルを使用してテキストを生成します。

### `generate_with_image(prompt: str, image_path: str) -> str`

指定されたモデルを使用して画像付きのテキストを生成します。

### `generate_json(prompt: str, schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]`

指定されたモデルを使用してJSONを生成します。

### `generate_with_image_json(prompt: str, image_path: str, schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]`

指定されたモデルを使用して画像付きのJSONを生成します。

### `set_api_key(model: str, api_key: str)`

指定されたモデルのAPIキーを設定します。

### `get_model() -> str`

使用中のモデル名を返します。
