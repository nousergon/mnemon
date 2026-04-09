"""Tests for the embedding pipeline — fragmentize, embed, embed_batch, embed_document."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import mnemon.embedder


@pytest.fixture(autouse=True)
def reset_model_singleton():
    """Reset the singleton model before each test."""
    mnemon.embedder._model = None
    yield
    mnemon.embedder._model = None


# ── fragmentize ───────────────────────────────────────────────────────────────


class TestFragmentize:
    def test_short_content_only_seq_zero(self):
        frags = mnemon.embedder.fragmentize("My Title", "Short content here.")
        assert len(frags) == 1
        assert frags[0]["seq"] == 0
        assert "title: My Title" in frags[0]["text"]
        assert "Short content here." in frags[0]["text"]

    def test_markdown_headers_create_sections(self):
        content = (
            "Introduction paragraph that is long enough to pass the 50 char filter easily.\n"
            "## Section One\n"
            "Content for section one that is definitely longer than fifty characters.\n"
            "## Section Two\n"
            "Content for section two that is also longer than fifty characters easily."
        )
        frags = mnemon.embedder.fragmentize("Doc", content)
        # seq=0 (full) + sections that pass the 50-char filter
        assert frags[0]["seq"] == 0
        section_frags = [f for f in frags if f["seq"] > 0]
        assert len(section_frags) >= 2
        assert all("title: Doc | section:" in f["text"] for f in section_frags)

    def test_max_five_sections(self):
        sections = "\n".join(
            f"## Section {i}\n{'X' * 60}" for i in range(10)
        )
        frags = mnemon.embedder.fragmentize("Many Sections", sections)
        section_frags = [f for f in frags if f["seq"] > 0]
        assert len(section_frags) <= 5

    def test_short_sections_filtered(self):
        content = (
            "## Short\nTiny.\n"
            "## Long Section\n" + "Y" * 100
        )
        frags = mnemon.embedder.fragmentize("Title", content)
        section_frags = [f for f in frags if f["seq"] > 0]
        # "Short\nTiny." is under 50 chars, should be filtered
        assert len(section_frags) == 1
        assert "Long Section" in section_frags[0]["text"]

    def test_full_text_truncated_to_2000_chars(self):
        long_content = "A" * 5000
        frags = mnemon.embedder.fragmentize("Title", long_content)
        assert len(frags[0]["text"]) <= 2000

    def test_section_text_truncated_to_1000_chars(self):
        content = "## Big Section\n" + "Z" * 2000
        frags = mnemon.embedder.fragmentize("Title", content)
        section_frags = [f for f in frags if f["seq"] > 0]
        assert len(section_frags) == 1
        # "title: Title | section: " prefix + 1000 chars max
        assert len(section_frags[0]["text"]) <= len("title: Title | section: ") + 1000 + 20


# ── embed ─────────────────────────────────────────────────────────────────────


class TestEmbed:
    def test_returns_float32_array(self):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.random.randn(384)])
        with patch("mnemon.embedder._get_model", return_value=mock_model):
            result = mnemon.embedder.embed("hello world")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (384,)

    def test_calls_model_with_list(self):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.random.randn(384)])
        with patch("mnemon.embedder._get_model", return_value=mock_model):
            mnemon.embedder.embed("test text")
        mock_model.embed.assert_called_once_with(["test text"])


# ── embed_batch ───────────────────────────────────────────────────────────────


class TestEmbedBatch:
    def test_returns_list_of_arrays(self):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.random.randn(384) for _ in range(3)])
        with patch("mnemon.embedder._get_model", return_value=mock_model):
            results = mnemon.embedder.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float32
            assert arr.shape == (384,)

    def test_calls_model_with_all_texts(self):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.random.randn(384), np.random.randn(384)])
        with patch("mnemon.embedder._get_model", return_value=mock_model):
            mnemon.embedder.embed_batch(["text1", "text2"])
        mock_model.embed.assert_called_once_with(["text1", "text2"])


# ── embed_document ────────────────────────────────────────────────────────────


class TestEmbedDocument:
    def test_saves_embeddings_and_flushes(self):
        mock_store = MagicMock()
        fake_emb = np.zeros(384, dtype=np.float32)
        with patch("mnemon.embedder.embed", return_value=fake_emb) as mock_embed:
            count = mnemon.embedder.embed_document(mock_store, "hash123", "Title", "Short content.")
        # Only seq=0 fragment for short content
        assert count == 1
        mock_store.save_embedding.assert_called_once_with("hash123", 0, fake_emb)
        mock_store.flush_vectors.assert_called_once()

    def test_multiple_fragments_all_saved(self):
        mock_store = MagicMock()
        fake_emb = np.zeros(384, dtype=np.float32)
        content = "## Section A\n" + "A" * 100 + "\n## Section B\n" + "B" * 100
        with patch("mnemon.embedder.embed", return_value=fake_emb):
            count = mnemon.embedder.embed_document(mock_store, "hash456", "Doc", content)
        # seq=0 + 2 sections
        assert count == 3
        assert mock_store.save_embedding.call_count == 3
        mock_store.flush_vectors.assert_called_once()

    def test_embed_called_per_fragment(self):
        mock_store = MagicMock()
        fake_emb = np.zeros(384, dtype=np.float32)
        with patch("mnemon.embedder.embed", return_value=fake_emb) as mock_embed:
            mnemon.embedder.embed_document(mock_store, "h", "T", "Short.")
        assert mock_embed.call_count == 1
        # Verify the text passed to embed contains the title
        call_text = mock_embed.call_args[0][0]
        assert "title: T" in call_text


# ── _get_model singleton ─────────────────────────────────────────────────────


class TestGetModel:
    def test_lazy_loads_model(self):
        mock_model = MagicMock()
        mock_fastembed = MagicMock()
        mock_fastembed.TextEmbedding.return_value = mock_model
        with patch.dict("sys.modules", {"fastembed": mock_fastembed}):
            result = mnemon.embedder._get_model()
        assert result is mock_model
        mock_fastembed.TextEmbedding.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")

    def test_singleton_returns_same_instance(self):
        fake_model = MagicMock()
        mnemon.embedder._model = fake_model
        result = mnemon.embedder._get_model()
        assert result is fake_model
