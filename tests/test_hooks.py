"""Tests for hook framework and extraction logic."""

import pytest

from mnemon.hooks.framework import is_noise, is_duplicate
from mnemon.hooks.session_extractor import (
    extract_with_regex,
    parse_observations,
)
from mnemon.hooks.handoff_generator import (
    generate_with_regex,
    parse_handoff,
)


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


class TestRegexExtractor:
    def test_extracts_decision(self):
        transcript = (
            "[user]: Let's discuss the database\n"
            "[assistant]: I decided to use PostgreSQL for the main store because of its JSON support."
        )
        obs = extract_with_regex(transcript)
        assert any(o["type"] == "decision" for o in obs)

    def test_extracts_observation(self):
        transcript = "[assistant]: I discovered that the API rate limit is 100 requests per minute."
        obs = extract_with_regex(transcript)
        assert any(o["type"] == "observation" for o in obs)

    def test_caps_at_five(self):
        transcript = "\n".join(
            f"[assistant]: We decided to use option {i} for the implementation."
            for i in range(10)
        )
        obs = extract_with_regex(transcript)
        assert len(obs) <= 5

    def test_empty_transcript(self):
        assert extract_with_regex("") == []

    def test_no_matches(self):
        transcript = "[user]: hello\n[assistant]: hi there"
        obs = extract_with_regex(transcript)
        assert len(obs) == 0


class TestLLMObservationParsing:
    def test_parses_single_observation(self):
        response = """<observation>
  <type>decision</type>
  <title>Use PostgreSQL for main store</title>
  <content>Decided to use PostgreSQL because of JSON support and strong ecosystem.</content>
</observation>"""
        results = parse_observations(response)
        assert len(results) == 1
        assert results[0]["type"] == "decision"
        assert results[0]["title"] == "Use PostgreSQL for main store"

    def test_parses_multiple_observations(self):
        response = """<observation>
  <type>decision</type>
  <title>Use PostgreSQL</title>
  <content>For JSON support.</content>
</observation>
<observation>
  <type>preference</type>
  <title>Single-line commands</title>
  <content>User prefers single-line shell commands.</content>
</observation>"""
        results = parse_observations(response)
        assert len(results) == 2

    def test_handles_none_response(self):
        response = "<none/>"
        results = parse_observations(response)
        assert len(results) == 0

    def test_handles_empty_fields(self):
        response = """<observation>
  <type>note</type>
  <title></title>
  <content></content>
</observation>"""
        results = parse_observations(response)
        assert len(results) == 0


class TestHandoffParsing:
    def test_parses_handoff(self):
        response = """<handoff>
  <title>Fixed auth bug and added tests</title>
  <summary>
  - Fixed JWT token validation in auth.py
  - Added 5 new tests for edge cases
  - Open: need to update docs
  </summary>
</handoff>"""
        result = parse_handoff(response)
        assert result is not None
        assert result["title"] == "Fixed auth bug and added tests"
        assert "JWT token" in result["summary"]

    def test_returns_none_for_missing_title(self):
        response = "<handoff><summary>stuff</summary></handoff>"
        assert parse_handoff(response) is None

    def test_returns_none_for_empty_content(self):
        response = "<handoff><title></title><summary></summary></handoff>"
        assert parse_handoff(response) is None


class TestRegexHandoff:
    def test_generates_handoff_from_transcript(self):
        transcript = (
            "[user]: Let's fix the deployment issue\n"
            "[assistant]: I'll look into it.\n"
            "[user]: Also update the config\n"
            "[assistant]: Done, I edited config.yaml"
        )
        result = generate_with_regex(transcript)
        assert result is not None
        assert "Topic" in result["summary"]

    def test_returns_none_for_short_transcript(self):
        transcript = "[user]: hi"
        result = generate_with_regex(transcript)
        assert result is None

    def test_includes_files_modified(self):
        transcript = (
            "[user]: Fix the bug\n"
            "[assistant]: I modified auth.py and updated config.yaml\n"
            "[user]: Great\n"
            "[assistant]: Done"
        )
        result = generate_with_regex(transcript)
        assert result is not None
        assert "Files touched" in result["summary"]
