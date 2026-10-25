import os
import sys
import json
import pytest

# Ensure the root directory is on the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    validate_inputs,
    build_roast_prompt,
    parse_roast_response,
    get_score_badge,
    format_markdown_report,
    MODEL_ID,
    SAMPLE_RESUMES,
)


class TestInputValidation:
    """Test suite for user input validation logic."""

    def test_missing_api_key(self):
        is_valid, err = validate_inputs("Some valid resume text that is quite long...", "Software Engineer", api_key=None)
        assert is_valid is False
        assert "API key is missing" in err

    def test_empty_role(self):
        is_valid, err = validate_inputs("Some valid resume text...", "", api_key="fake-key")
        assert is_valid is False
        assert "Target role cannot be empty" in err

    def test_whitespace_role(self):
        is_valid, err = validate_inputs("Some valid resume text...", "   ", api_key="fake-key")
        assert is_valid is False
        assert "Target role cannot be empty" in err

    def test_empty_resume(self):
        is_valid, err = validate_inputs("", "Software Engineer", api_key="fake-key")
        assert is_valid is False
        assert "Please paste your resume" in err

    def test_too_short_resume_characters(self):
        is_valid, err = validate_inputs("Short resume text.", "Software Engineer", api_key="fake-key")
        assert is_valid is False
        assert "too brief" in err

    def test_too_few_words_resume(self):
        # 5 words, but well over 80 characters
        long_words = "ComprehensiveEngineeringMethodology " * 5
        is_valid, err = validate_inputs(long_words, "Software Engineer", api_key="fake-key")
        assert is_valid is False
        assert "words" in err

    def test_valid_inputs_success(self):
        valid_resume = (
            "Jane Doe\n"
            "Software Developer with 2 years of experience building web applications in Python and React.\n"
            "Engineered RESTful APIs with FastAPI and deployed containerized services on Docker."
        )
        is_valid, err = validate_inputs(valid_resume, "Full Stack Developer", api_key="fake-key-12345")
        assert is_valid is True
        assert err is None


class TestPromptConstruction:
    """Test suite for Gemma prompt construction and integrity instructions."""

    def test_prompt_contains_model_target_role(self):
        resume = SAMPLE_RESUMES["Weak Junior Dev"]
        target_role = "Senior Cloud Architect"
        prompt = build_roast_prompt(resume, target_role, roast_intensity="Extra Crispy")

        assert target_role in prompt
        assert "Alex Johnson" in prompt
        assert "CRITICAL INTEGRITY INSTRUCTIONS" in prompt

    def test_prompt_forbids_hallucinated_metrics(self):
        prompt = build_roast_prompt("Sample resume content", "Backend Engineer")
        assert "NEVER fabricate, invent, or hallucinate" in prompt
        assert "[X%]" in prompt
        assert "single valid JSON object" in prompt

    def test_prompt_intensity_variations(self):
        p_gentle = build_roast_prompt("Sample text", "DevOps", roast_intensity="Gentle Simmer")
        p_crispy = build_roast_prompt("Sample text", "DevOps", roast_intensity="Extra Crispy")

        assert "Gentle Simmer" in p_gentle or "encouraging" in p_gentle
        assert "Extra Crispy" in p_crispy or "Brutally witty" in p_crispy


class TestResponseParsing:
    """Test suite for JSON response parsing and safety fallbacks."""

    def test_parse_valid_clean_json(self):
        mock_data = {
            "overall_score": 68,
            "score_breakdown": {
                "clarity": 75,
                "skills": 70,
                "project_descriptions": 60,
                "measurable_impact": 50,
            },
            "roast_punchline": "Your resume reads like an IKEA manual without pictures.",
            "verdict_summary": "Great raw potential, but deeply needs quantifiable impact.",
            "weaknesses": [
                {"title": "Weak Impact", "explanation": "No numbers anywhere.", "impact": "Rejected by ATS."}
            ],
            "rewritten_bullets": [
                {"original": "Made website.", "rewritten": "Architected web portal reducing load time by [X%].", "why": "Adds verbs."}
            ],
            "prioritized_improvements": [
                {"priority": 1, "area": "Metrics", "action": "Add percentages."}
            ],
        }

        parsed = parse_roast_response(json.dumps(mock_data))
        assert parsed["overall_score"] == 68
        assert parsed["score_breakdown"]["clarity"] == 75
        assert parsed["roast_punchline"] == "Your resume reads like an IKEA manual without pictures."
        assert len(parsed["weaknesses"]) == 1
        assert parsed["rewritten_bullets"][0]["rewritten"].startswith("Architected")

    def test_parse_markdown_fenced_json(self):
        raw_text = """```json
{
  "overall_score": 82,
  "score_breakdown": {
    "clarity": 85,
    "skills": 90,
    "project_descriptions": 80,
    "measurable_impact": 75
  },
  "roast_punchline": "A surprisingly sharp resume!",
  "verdict_summary": "Solid candidate with strong projects.",
  "weaknesses": [
    {"title": "Minor formatting", "explanation": "Spacing", "impact": "Slight distraction"}
  ],
  "rewritten_bullets": [],
  "prioritized_improvements": []
}
```"""
        parsed = parse_roast_response(raw_text)
        assert parsed["overall_score"] == 82
        assert parsed["score_breakdown"]["skills"] == 90
        assert parsed["roast_punchline"] == "A surprisingly sharp resume!"

    def test_parse_invalid_json_fallback(self):
        malformed = "Here is my roast:\nSorry, I could not generate JSON output today because of a glitch."
        parsed = parse_roast_response(malformed)

        # Fallback dictionary should gracefully supply valid default structure
        assert "overall_score" in parsed
        assert isinstance(parsed["overall_score"], int)
        assert "score_breakdown" in parsed
        assert len(parsed["weaknesses"]) == 3
        assert len(parsed["rewritten_bullets"]) >= 1
        assert len(parsed["prioritized_improvements"]) >= 1


class TestScoreBadges:
    """Test suite for score grading and badge mapping."""

    def test_badge_high_score(self):
        badge = get_score_badge(95)
        assert badge["grade"] == "A+"
        assert "Interview Magnet" in badge["status"]

    def test_badge_medium_score(self):
        badge = get_score_badge(72)
        assert badge["grade"] == "B"
        assert "Decent Foundation" in badge["status"]

    def test_badge_low_score(self):
        badge = get_score_badge(45)
        assert badge["grade"] == "D"
        assert "Extra Crispy" in badge["flame"]

    def test_badge_failing_score(self):
        badge = get_score_badge(30)
        assert badge["grade"] == "F"
        assert "Incinerated" in badge["flame"]


class TestMarkdownReportFormatting:
    """Test suite for markdown export report generation."""

    def test_report_contains_all_key_sections(self):
        mock_result = {
            "overall_score": 65,
            "score_breakdown": {
                "clarity": 70,
                "skills": 65,
                "project_descriptions": 60,
                "measurable_impact": 50,
            },
            "roast_punchline": "You know 20 frameworks but built zero metrics.",
            "verdict_summary": "Focus on showing what you achieved, not just what was installed.",
            "weaknesses": [
                {"title": "No metrics", "explanation": "Zero numbers.", "impact": "No proof."}
            ],
            "rewritten_bullets": [
                {"original": "Worked on API.", "rewritten": "Designed REST API in Go with [X%] latency reduction.", "why": "Better."}
            ],
            "prioritized_improvements": [
                {"priority": 1, "area": "Quantification", "action": "Add real numbers."}
            ],
        }

        report = format_markdown_report(mock_result, "Backend Engineer")
        assert "# 🔥 Resume Roast AI Report" in report
        assert "**Target Role:** Backend Engineer" in report
        assert MODEL_ID in report
        assert "65/100" in report
        assert "You know 20 frameworks but built zero metrics." in report
        assert "Designed REST API in Go" in report
        assert "Notice: This score and roast is an AI-generated estimate" in report
