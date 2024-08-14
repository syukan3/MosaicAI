from .client import MosaicAI
from .exceptions import MosaicAIError, ModelNotSupportedError, APIKeyNotFoundError, InvalidJSONSchemaError

__all__ = [
    'MosaicAI',
    'MosaicAIError',
    'ModelNotSupportedError',
    'APIKeyNotFoundError',
    'InvalidJSONSchemaError'
]

__version__ = "0.1.1"
