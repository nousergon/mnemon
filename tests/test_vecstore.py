"""Tests for the vector store."""

import os
import tempfile

import numpy as np
import pytest

from mnemon.vecstore import VecStore


@pytest.fixture
def vecstore():
    fd, path = tempfile.mkstemp(suffix=".vec")
    os.close(fd)
    os.unlink(path)
    vs = VecStore(path, dim=4)
    yield vs
    for ext in ("", ".npz"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


class TestVecStore:
    def test_set_and_size(self, vecstore):
        vecstore.set("a_0", np.array([1, 0, 0, 0], dtype=np.float32))
        assert vecstore.size() == 1

    def test_has(self, vecstore):
        vecstore.set("a_0", np.array([1, 0, 0, 0], dtype=np.float32))
        assert vecstore.has("a_0")
        assert not vecstore.has("b_0")

    def test_delete(self, vecstore):
        vecstore.set("a_0", np.array([1, 0, 0, 0], dtype=np.float32))
        assert vecstore.delete("a_0")
        assert vecstore.size() == 0
        assert not vecstore.delete("a_0")

    def test_search_cosine(self, vecstore):
        vecstore.set("a_0", np.array([1, 0, 0, 0], dtype=np.float32))
        vecstore.set("b_0", np.array([0, 1, 0, 0], dtype=np.float32))
        vecstore.set("c_0", np.array([0.9, 0.1, 0, 0], dtype=np.float32))

        results = vecstore.search(np.array([1, 0, 0, 0], dtype=np.float32), k=3)
        assert len(results) == 3
        # "a_0" should be most similar (exact match)
        assert results[0]["id"] == "a_0"
        assert results[0]["similarity"] == pytest.approx(1.0, abs=0.01)
        # "c_0" should be second (close to query)
        assert results[1]["id"] == "c_0"

    def test_search_empty(self, vecstore):
        results = vecstore.search(np.array([1, 0, 0, 0], dtype=np.float32))
        assert results == []

    def test_persistence(self, vecstore):
        vecstore.set("a_0", np.array([1, 0, 0, 0], dtype=np.float32))
        vecstore.set("b_0", np.array([0, 1, 0, 0], dtype=np.float32))
        vecstore.save()

        # Load from disk
        vs2 = VecStore(vecstore.file_path, dim=4)
        assert vs2.size() == 2
        assert vs2.has("a_0")
        results = vs2.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
        assert results[0]["id"] == "a_0"

    def test_wrong_dim_ignored(self, vecstore):
        vecstore.set("a_0", np.array([1, 0, 0, 0], dtype=np.float32))
        vecstore.save()

        vs2 = VecStore(vecstore.file_path, dim=8)
        assert vs2.size() == 0  # Should ignore mismatched vectors

    def test_dim_validation(self, vecstore):
        with pytest.raises(ValueError):
            vecstore.set("a_0", np.array([1, 0, 0], dtype=np.float32))
