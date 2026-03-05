"""llms.py – unified LLM interface for de-rag.

Provides a common ``BaseLLM`` ABC so that generation is decoupled from
the underlying backend.

Supported backends
------------------
- CohereLLM  – wraps the Cohere Chat API (requires ``cohere``)
- OllamaLLM  – wraps the Ollama local API (requires a running Ollama server)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from de_rag.classes import RetrievalResult
from de_rag.logger import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    """Abstract base class for all LLM backends."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate an answer for *query* grounded in *context* documents.

        Parameters
        ----------
        query:
            The user question.
        context:
            Retrieved documents to ground the answer in.
        system_prompt:
            Optional override for the default system instruction.
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature (0 = deterministic).

        Returns
        -------
        str
            The model's response text.
        """


# ── Cohere backend ───────────────────────────────────────────────────────────────────────────────

class CohereLLM(BaseLLM):
    """LLM backed by the Cohere Chat API.

    Parameters
    ----------
    api_key:
        Cohere API key.  If ``None``, the ``COHERE_API_KEY`` environment
        variable is used automatically by the ``cohere`` client.
    model:
        Cohere generation model name (e.g. ``"command-r-plus"``).

    Notes
    -----
    Requires ``pip install cohere``.
    Retrieved documents are passed via Cohere's native ``documents`` field so
    the model can cite them directly.
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful assistant. Answer the user's question based only "
        "on the provided context documents. If the answer is not in the "
        "context, say you don't know."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "command-r-plus",
    ) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Package 'cohere' is not installed. "
                "Run 'pip install cohere' to use CohereLLM."
            )
        import httpx

        client = httpx.Client(verify=False)
        self._client = (
            cohere.Client(api_key, httpx_client=client)
            if api_key
            else cohere.Client(httpx_client=client)
        )
        self._model = model
        logger.info("Initialized CohereLLM with model '%s'", model)

    def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        preamble = system_prompt or self._DEFAULT_SYSTEM
        documents = [
            {"id": str(i), "text": r.doc.text}
            for i, r in enumerate(context)
        ]
        logger.debug(
            "CohereLLM.generate: model=%s, docs=%d, query=%r",
            self._model,
            len(documents),
            query[:80],
        )
        response = self._client.chat(
            model=self._model,
            preamble=preamble,
            documents=documents,
            message=query,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.text


# ── Ollama backend ───────────────────────────────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    """LLM backed by a locally running Ollama server.

    Parameters
    ----------
    model:
        Ollama model tag (e.g. ``"llama3"``, ``"mistral"``).
    base_url:
        Ollama server base URL. Defaults to ``http://localhost:11434``.

    Notes
    -----
    Requires ``pip install requests`` and a running Ollama instance.
    Context documents are injected into the user message so any model works
    without native document support.
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful assistant. Answer the user's question based only "
        "on the provided context documents. If the answer is not in the "
        "context, say you don't know."
    )

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Package 'requests' is not installed. "
                "Run 'pip install requests' to use OllamaLLM."
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._requests = requests
        logger.info(
            "Initialized OllamaLLM with model '%s' at %s", model, self._base_url
        )

    def _build_messages(
        self,
        query: str,
        context: List[RetrievalResult],
        system_prompt: Optional[str],
    ) -> List[dict]:
        system = system_prompt or self._DEFAULT_SYSTEM
        context_text = "\n\n".join(
            f"[{i + 1}] {r.doc.text}" for i, r in enumerate(context)
        )
        user_content = f"Context:\n\n{context_text}\n\nQuestion: {query}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        query: str,
        context: List[RetrievalResult],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = self._build_messages(query, context, system_prompt)
        payload = {
            "model": self._model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        logger.debug(
            "OllamaLLM.generate: model=%s, docs=%d, query=%r",
            self._model,
            len(context),
            query[:80],
        )
        resp = self._requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
