import pytest
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import (
    remove_html_and_urls,
    normalize_arabic,
    filter_characters,
    clean_whitespace,
    is_valid_length,
    pipeline_step,
)
def test_removes_http_url():
    assert "text" in remove_html_and_urls("text https://example.com end")
    assert "https://" not in remove_html_and_urls("text https://example.com")

def test_removes_www_url():
    result = remove_html_and_urls("visit www.example.com today")
    assert "www." not in result

def test_removes_html_tags():
    result = remove_html_and_urls("<p>النص هنا</p>")
    assert "<p>" not in result
    assert "النص هنا" in result

def test_removes_nested_html():
    result = remove_html_and_urls("<div><span>text</span></div>")
    assert "<" not in result
    assert "text" in result

def test_empty_string_returns_empty():
    assert remove_html_and_urls("") == ""

def test_no_html_unchanged():
    text = "النص العربي بدون html"
    assert remove_html_and_urls(text) == text

@pytest.mark.parametrize("input_char,expected", [
    ("إ", "ا"),
    ("أ", "ا"),
    ("آ", "ا"),
])
def test_normalizes_hamza_variants(input_char, expected):
    result = normalize_arabic(input_char)
    assert result == expected

def test_removes_tatweel():
    result = normalize_arabic("مـرحـبا")
    assert "\u0640" not in result
    assert "مرحبا" in result

def test_removes_diacritics():
    result = normalize_arabic("مَرْحَبًا")
    assert "َ" not in result
    assert "ْ" not in result

def test_normalizes_alef_maqsura():
    result = normalize_arabic("على")
    assert "ى" not in result
    assert "علي" in result

def test_normalizes_taa_marbuta():
    result = normalize_arabic("مدرسة")
    assert "ة" not in result

def test_collapses_repeated_chars():
    result = normalize_arabic("لاااااا")
    assert len(result) <= 4

def test_normalizes_lamalef_ligatures():
    result = normalize_arabic("ﻻ")
    assert "لا" in result

def test_empty_string():
    assert normalize_arabic("") == ""

def test_keeps_arabic():
    result = filter_characters("النص العربي")
    assert "النص" in result

def test_keeps_latin():
    result = filter_characters("MLOps project")
    assert "MLOps" in result

def test_keeps_digits():
    result = filter_characters("123 ١٢٣")
    assert "123" in result

def test_keeps_french_accents():
    result = filter_characters("café résumé")
    assert "café" in result

def test_removes_special_chars():
    result = filter_characters("text @#$% text")
    assert "@" not in result
    assert "#" not in result

def test_empty_string():
    assert filter_characters("") == ""

def test_collapses_multiple_spaces():
    result = clean_whitespace("النص    هنا")
    assert "  " not in result

def test_removes_tabs():
    result = clean_whitespace("النص\tهنا")
    assert "\t" not in result

def test_removes_newlines():
    result = clean_whitespace("النص\nهنا")
    assert "\n" not in result

def test_strips_leading_trailing():
    result = clean_whitespace("  النص  ")
    assert result == result.strip()

def test_empty_string():
    assert clean_whitespace("") == ""

def test_only_whitespace():
    assert clean_whitespace("   ") == ""

def test_valid_length_passes():
    assert is_valid_length("النص الكافي للتدريب", min_length=10) is True

def test_too_short_fails():
    assert is_valid_length("قصير", min_length=10) is False

def test_exact_min_length_passes():
    text = "ن" * 10
    assert is_valid_length(text, min_length=10) is True

def test_empty_string_fails():
    assert is_valid_length("", min_length=10) is False

def test_whitespace_only_fails():
    assert is_valid_length("     ", min_length=10) is False

def test_pipeline_removes_html_and_normalizes():
    result = pipeline_step("<p>إعلان جديد</p>")
    assert "<p>" not in result
    assert "إ" not in result  

def test_pipeline_handles_empty():
    assert pipeline_step("") == ""

def test_pipeline_handles_url_with_arabic():
    result = pipeline_step("تفاصيل على https://example.com اضغط هنا")
    assert "https://" not in result
    assert "تفاصيل" in result

def test_pipeline_is_deterministic():
    text = "إعلان جديد عن برنامج تعليمي"
    assert pipeline_step(text) == pipeline_step(text)

@pytest.mark.parametrize("text", [
    "<div>النص</div>",
    "https://hespress.com النص",
    "النص مـع تطويل",
    "إأآ variants",
    "النص   مع   مسافات",
])
def test_pipeline_handles_various_inputs(text):
    result = pipeline_step(text)
    assert isinstance(result, str)