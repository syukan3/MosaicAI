# MosaicAIの基本例外クラス
class MosaicAIError(Exception):
    """Base exception class for MosaicAI"""

# サポートされていないモデルが要求されたときに発生する例外
class ModelNotSupportedError(MosaicAIError):
    """Raised when an unsupported model is requested"""

# 要求されたモデルのAPIキーが見つからないときに発生する例外
class APIKeyNotFoundError(MosaicAIError):
    """Raised when an API key is not found for a requested model"""

# 無効なJSONスキーマが提供されたときに発生する例外
class InvalidJSONSchemaError(MosaicAIError):
    """Raised when an invalid JSON schema is provided"""