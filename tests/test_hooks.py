"""Tests for hook framework and extraction logic."""

import pytest

from mnemon.hooks.framework import is_noise, is_duplicate
from mnemon.hooks.session_extractor import extract_observations


class TestNoiseFilter:
    def test_empty_is_noise(self):
        assert is_noise("")

    def test_short_is_noise(self):
        assert is_noise("hi")

    def test_greeting_is_noise(self):
        assert is_noise("hello")
        assert is_noise("thanks!")
        assert is_noise("ok")

    def test_slash_command_is_noise(self):
        assert is_noise("/help")
        assert is_noise("/clear")

    def test_real_prompt_is_not_noise(self):
        assert not is_noise("how does the deployment pipeline work?")
        assert not is_noise("fix the bug in auth.py")

    def test_single_letter_is_noise(self):
        assert is_noise("y")
        assert is_noise("n")


class TestSessionExtractor:
    def test_extracts_decision(self):
        transcript = "[user]: Let's discuss the database\n[assistant]: I decided to use PostgreSQL for the main store because of its JSON support."
        obs = extract_observations(transcript)
        assert any(o["type"] == "decision" for o in obs)

    def test_extracts_observation(self):
        transcript = "[assistant]: I discovered that the API rate limit is 100 requests per minute."
        obs = extract_observations(transcript)
        assert any(o["type"] == "observation" for o in obs)

    def test_caps_at_five(self):
        transcript = "\n".join(
            f"[assistant]: We decided to use option {i} for the implementation."
            for i in range(10)
        )
        obs = extract_observations(transcript)
        assert len(obs) <= 5

    def test_empty_transcript(self):
        assert extract_observations("") == []

    def test_no_matches(self):
        transcript = "[user]: hello\n[assistant]: hi there"
        obs = extract_observations(transcript)
        assert len(obs) == 0
