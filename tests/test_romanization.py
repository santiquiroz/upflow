from app.services.romanization import (
    contains_japanese,
    romanize_segments,
    romanize_text,
)
from app.services.subtitles import TranscriptSegment, WordSpan


def test_contains_japanese_detects_kanji_and_katakana():
    assert contains_japanese("鉄壁") is True
    assert contains_japanese("カラオケ") is True


def test_contains_japanese_false_for_latin_and_empty():
    assert contains_japanese("hello world") is False
    assert contains_japanese("") is False


def test_romanize_text_returns_english_unchanged():
    assert romanize_text("THE FIRST TAKE") == "THE FIRST TAKE"


def test_romanize_text_converts_japanese_to_ascii():
    result = romanize_text("鉄壁")

    assert result != "鉄壁"
    assert result.isascii()
    assert len(result) > 0


def test_romanize_text_single_kanji_returns_ascii():
    assert romanize_text("空").isascii()


def test_romanize_segments_preserves_timing_exactly():
    segment = TranscriptSegment(
        start=1.25,
        end=3.5,
        text="鉄壁だ",
        words=(
            WordSpan(word="鉄壁", start=1.25, end=2.0),
            WordSpan(word="だ", start=2.0, end=3.5),
        ),
    )

    result = romanize_segments([segment])[0]

    assert result.start == 1.25
    assert result.end == 3.5
    assert [(w.start, w.end) for w in result.words] == [(1.25, 2.0), (2.0, 3.5)]


def test_romanize_segments_leaves_english_segment_identical():
    segment = TranscriptSegment(
        start=1.25,
        end=3.5,
        text="hello",
        words=(WordSpan(word="hello", start=1.25, end=3.5),),
    )

    result = romanize_segments([segment])[0]

    assert result.text == "hello"
    assert result.words[0].word == "hello"


def test_romanize_segments_converts_japanese_words_to_ascii():
    segment = TranscriptSegment(
        start=1.25,
        end=3.5,
        text="鉄壁",
        words=(WordSpan(word="鉄壁", start=1.25, end=3.5),),
    )

    result = romanize_segments([segment])[0]

    assert result.text.isascii()
    assert result.words[0].word.isascii()


def test_romanize_segments_returns_new_list_without_mutating_input():
    segment = TranscriptSegment(start=0.0, end=1.0, text="鉄壁")
    originales = [segment]

    result = romanize_segments(originales)

    assert result is not originales
    assert originales[0].text == "鉄壁"


def test_romanize_segments_empty_list():
    assert romanize_segments([]) == []
