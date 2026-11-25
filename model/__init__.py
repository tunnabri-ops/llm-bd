"""LLM BD model package.

Imports are lazy: the tokenizer works without torch installed, the model
needs torch (see requirements.txt).
"""
try:
    from .model import LLM, ModelConfig, count_parameters
except ImportError:  # torch not installed yet — tokenizer-only use still works
    LLM = ModelConfig = count_parameters = None  # type: ignore

from .tokenizer import BanglaTokenizer, SPECIAL_TOKENS

__all__ = ["LLM", "ModelConfig", "count_parameters", "BanglaTokenizer", "SPECIAL_TOKENS"]
