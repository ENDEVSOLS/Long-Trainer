"""Model and Embedding Factories for LongTrainer."""

from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

def get_llm(provider: str, model_name: str, **kwargs: Any) -> BaseChatModel:
    """Instantiate a ChatModel based on the provider string."""
    provider = provider.lower()
    
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model_name=model_name, **kwargs)
        except ImportError:
            raise ImportError("Please install langchain-openai to use OpenAI models: pip install langchain-openai")
            
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model_name=model_name, **kwargs)
        except ImportError:
            raise ImportError("Please install langchain-anthropic to use Anthropic models: pip install langchain-anthropic")
            
    elif provider in {"google", "gemini"}:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, **kwargs)
        except ImportError:
            raise ImportError("Please install langchain-google-genai to use Google models: pip install langchain-google-genai")
            
    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name, **kwargs)
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(model=model_name, **kwargs)
            except ImportError:
                raise ImportError("Please install langchain-ollama to use Ollama models: pip install langchain-ollama")
                
    elif provider == "huggingface":
        try:
            from langchain_huggingface import ChatHuggingFace
            from langchain_huggingface import HuggingFaceEndpoint
            llm = HuggingFaceEndpoint(repo_id=model_name, **kwargs)
            return ChatHuggingFace(llm=llm)
        except ImportError:
            raise ImportError("Please install langchain-huggingface to use HuggingFace models: pip install langchain-huggingface")
            
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embedding_model(provider: str, model_name: str, **kwargs: Any) -> Embeddings:
    """Instantiate an Embeddings model based on the provider string."""
    provider = provider.lower()
    
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            # OpenAIEmbeddings typically takes `model` instead of `model_name` for new packages, but backwards compatibility handles model_name too. We'll pass `model`.
            return OpenAIEmbeddings(model=model_name, **kwargs)
        except ImportError:
            raise ImportError("Please install langchain-openai to use OpenAI embeddings: pip install langchain-openai")
            
    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        except ImportError:
            raise ImportError("Please install langchain-huggingface to use HuggingFace embeddings: pip install langchain-huggingface sentence-transformers")
            
    elif provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(model=model_name, **kwargs)
        except ImportError:
            raise ImportError("Please install langchain-cohere to use Cohere embeddings: pip install langchain-cohere")
            
    elif provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model_name, **kwargs)
        except ImportError:
            try:
                from langchain_community.embeddings import OllamaEmbeddings
                return OllamaEmbeddings(model=model_name, **kwargs)
            except ImportError:
                raise ImportError("Please install langchain-ollama to use Ollama embeddings: pip install langchain-ollama")
                
    else:
        raise ValueError(f"Unsupported Embedding provider: {provider}")
