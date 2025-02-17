import pytest
from errant.converter import convert_m2_to_text

def test_deletion():
    m2_str = """S It 's difficult answer at the question " what are you going to do in the future ? " if the only one who has to know it is in two minds .
A 3 3|||M:VERB:FORM|||to|||REQUIRED|||-NONE-|||0
A 4 5|||U:PREP||||||REQUIRED|||-NONE-|||0"""

    expected = "It 's difficult to answer the question \" what are you going to do in the future ? \" if the only one who has to know it is in two minds ."
    result = convert_m2_to_text(m2_str)
    assert result['corrected'] == expected

def test_m2_with_no_edits():
    m2_str = """S This sentence has no errors.
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"""
    expected = "This sentence has no errors."
    result = convert_m2_to_text(m2_str)
    assert result['corrected'] == expected

def test_m2_with_multiple_edits():
    m2_str = """S He have a lot of book in his room.
A 1 2|||R:VERB:SVA|||has|||REQUIRED|||-NONE-|||0
A 5 6|||R:NOUN:NUM|||books|||REQUIRED|||-NONE-|||0"""
    
    expected = "He has a lot of books in his room."
    result = convert_m2_to_text(m2_str)
    assert result['corrected'] == expected

def test_m2_with_empty_string():
    with pytest.raises(ValueError):
        convert_m2_to_text("")

def test_m2_with_invalid_format():
    with pytest.raises(ValueError):
        convert_m2_to_text("Invalid M2 format")
