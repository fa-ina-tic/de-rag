"""Tests for de_rag.llms (CohereLLM, OllamaLLM)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from de_rag.classes import Document, RetrievalResult
from de_rag.llms import CohereLLM, OllamaLLM


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_result(text, score=0.9, source="test"):
    import numpy as np
    doc = Document(id="d1", text=text, embedding=np.zeros(4), doc_type="chunk")
    return RetrievalResult(doc=doc, score=score, source=source)


@pytest.fixture
def context():
    return [
        _make_result("Paris is the capital of France."),
        _make_result("The Eiffel Tower is in Paris."),
    ]


# ── CohereLLM ──────────────────────────────────────────────────────────────────

class TestCohereLLM:

    @pytest.fixture
    def cohere_client(self):
        client = MagicMock()
        response = MagicMock()
        response.text = "Paris is the capital of France."
        client.chat.return_value = response
        return client

    @pytest.fixture
    def llm(self, cohere_client):
        llm = CohereLLM.__new__(CohereLLM)
        llm._client = cohere_client
        llm._model = "command-r-plus"
        return llm

    def test_generate_returns_string(self, llm, context):
        result = llm.generate("What is the capital of France?", context)
        assert isinstance(result, str)

    def test_generate_calls_chat(self, llm, cohere_client, context):
        llm.generate("query", context)
        cohere_client.chat.assert_called_once()

    def test_generate_passes_model(self, llm, cohere_client, context):
        llm.generate("query", context)
        call_kwargs = cohere_client.chat.call_args.kwargs
        assert call_kwargs.get("model") == "command-r-plus"

    def test_generate_passes_documents(self, llm, cohere_client, context):
        llm.generate("query", context)
        call_kwargs = cohere_client.chat.call_args.kwargs
        docs = call_kwargs.get("documents")
        assert len(docs) == len(context)
        for doc in docs:
            assert "id" in doc
            assert "text" in doc

    def test_generate_passes_message(self, llm, cohere_client, context):
        query = "What is the capital?"
        llm.generate(query, context)
        call_kwargs = cohere_client.chat.call_args.kwargs
        assert call_kwargs.get("message") == query

    def test_generate_uses_default_system_prompt(self, llm, cohere_client, context):
        llm.generate("query", context)
        call_kwargs = cohere_client.chat.call_args.kwargs
        preamble = call_kwargs.get("preamble")
        assert preamble == CohereLLM._DEFAULT_SYSTEM

    def test_generate_uses_custom_system_prompt(self, llm, cohere_client, context):
        custom = "Answer only in French."
        llm.generate("query", context, system_prompt=custom)
        call_kwargs = cohere_client.chat.call_args.kwargs
        assert call_kwargs.get("preamble") == custom

    def test_generate_passes_max_tokens(self, llm, cohere_client, context):
        llm.generate("query", context, max_tokens=256)
        call_kwargs = cohere_client.chat.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 256

    def test_generate_passes_temperature(self, llm, cohere_client, context):
        llm.generate("query", context, temperature=0.7)
        call_kwargs = cohere_client.chat.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.7

    def test_generate_empty_context(self, llm, cohere_client):
        llm.generate("query", [])
        call_kwargs = cohere_client.chat.call_args.kwargs
        assert call_kwargs.get("documents") == []

    def test_generate_document_ids_are_sequential(self, llm, cohere_client, context):
        llm.generate("query", context)
        docs = cohere_client.chat.call_args.kwargs.get("documents")
        for i, doc in enumerate(docs):
            assert doc["id"] == str(i)


# ── OllamaLLM ──────────────────────────────────────────────────────────────────

class TestOllamaLLM:

    @pytest.fixture
    def mock_requests(self):
        requests = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"message": {"content": "The answer is Paris."}}
        resp.raise_for_status.return_value = None
        requests.post.return_value = resp
        return requests

    @pytest.fixture
    def llm(self, mock_requests):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm._model = "llama3"
        llm._base_url = "http://localhost:11434"
        llm._requests = mock_requests
        return llm

    def test_generate_returns_string(self, llm, context):
        result = llm.generate("What is the capital?", context)
        assert isinstance(result, str)

    def test_generate_posts_to_correct_url(self, llm, mock_requests, context):
        llm.generate("query", context)
        url = mock_requests.post.call_args.args[0]
        assert url == "http://localhost:11434/api/chat"

    def test_generate_sends_correct_model(self, llm, mock_requests, context):
        llm.generate("query", context)
        payload = mock_requests.post.call_args.kwargs.get("json")
        assert payload["model"] == "llama3"

    def test_generate_stream_false(self, llm, mock_requests, context):
        llm.generate("query", context)
        payload = mock_requests.post.call_args.kwargs.get("json")
        assert payload["stream"] is False

    def test_generate_passes_temperature(self, llm, mock_requests, context):
        llm.generate("query", context, temperature=0.5)
        payload = mock_requests.post.call_args.kwargs.get("json")
        assert payload["options"]["temperature"] == 0.5

    def test_generate_passes_max_tokens(self, llm, mock_requests, context):
        llm.generate("query", context, max_tokens=100)
        payload = mock_requests.post.call_args.kwargs.get("json")
        assert payload["options"]["num_predict"] == 100

    def test_generate_calls_raise_for_status(self, llm, mock_requests, context):
        llm.generate("query", context)
        mock_requests.post.return_value.raise_for_status.assert_called_once()

    # _build_messages
    def test_build_messages_has_system_and_user(self, llm, context):
        msgs = llm._build_messages("What?", context, system_prompt=None)
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert "user" in roles

    def test_build_messages_uses_default_system(self, llm, context):
        msgs = llm._build_messages("q", context, system_prompt=None)
        system_msg = next(m for m in msgs if m["role"] == "system")
        assert system_msg["content"] == OllamaLLM._DEFAULT_SYSTEM

    def test_build_messages_uses_custom_system(self, llm, context):
        msgs = llm._build_messages("q", context, system_prompt="Custom prompt.")
        system_msg = next(m for m in msgs if m["role"] == "system")
        assert system_msg["content"] == "Custom prompt."

    def test_build_messages_includes_context_text(self, llm, context):
        msgs = llm._build_messages("q", context, system_prompt=None)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "Paris is the capital of France." in user_msg["content"]
        assert "The Eiffel Tower is in Paris." in user_msg["content"]

    def test_build_messages_includes_query(self, llm, context):
        query = "What is the capital of France?"
        msgs = llm._build_messages(query, context, system_prompt=None)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert query in user_msg["content"]

    def test_base_url_trailing_slash_stripped(self):
        requests = MagicMock()
        with patch.dict("sys.modules", {"requests": requests}):
            llm = OllamaLLM.__new__(OllamaLLM)
            llm._model = "llama3"
            llm._base_url = "http://localhost:11434/"
            llm._base_url = llm._base_url.rstrip("/")
            llm._requests = requests
        assert not llm._base_url.endswith("/")
