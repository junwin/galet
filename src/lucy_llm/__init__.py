from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .adapter_interface import LLMAdapter

try:
    from .openai_responses import OpenAIResponsesApi
    from .openai_responses_adapter import OpenAIResponsesAdapter
except Exception:
    # openai and related adapters are optional for test environments; avoid
    # failing import when the 'openai' package is not installed.
    OpenAIResponsesApi = None
    OpenAIResponsesAdapter = None

try:
    from .mistral_api import MistralApi
    from .mistral_responses_adapter import MistralResponsesAdapter
except Exception:
    MistralApi = None
    MistralResponsesAdapter = None

try:
    from .ollama_api import OllamaApi
except Exception:
    OllamaApi = None

try:
    from .gemini_api import GeminiApi
except Exception:
    GeminiApi = None

__all__ = [
    "LLMApi",
    "LLMAdapter",
    "LLMResponse",
    "LLMUsage",
    "ToolCall",
    "OpenAIResponsesApi",
    "OpenAIResponsesAdapter",
    "MistralApi",
    "MistralResponsesAdapter",
    "OllamaApi",
    "GeminiApi",
]
