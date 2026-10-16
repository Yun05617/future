from speech_emotion.traditional_chinese import to_traditional_chinese
import pytest


def test_to_traditional_chinese_converts_simplified_text():
    assert to_traditional_chinese("我失恋了") == "我失戀了"


@pytest.mark.parametrize("text, expected", [
    ("语音识别，学习与情绪", "語音識別，學習與情緒"),
    ("繁體中文 ABC 123。\n你好！", "繁體中文 ABC 123。\n你好！"),
    ("你好 😀，听得到吗？", "你好 😀，聽得到嗎？"),
])
def test_converter_preserves_mixed_text(text, expected):
    assert to_traditional_chinese(text) == expected


@pytest.mark.parametrize("text", ["", "   ", None])
def test_converter_rejects_empty_or_non_text(text):
    from speech_emotion.traditional_chinese import TraditionalChineseError

    with pytest.raises(TraditionalChineseError):
        to_traditional_chinese(text)
