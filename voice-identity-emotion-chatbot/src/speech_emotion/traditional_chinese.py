"""Offline Chinese conversion using Microsoft NLS or Apple Foundation."""

import ctypes
import sys


class TraditionalChineseError(ValueError):
    """Raised when the operating system's local converter cannot be used."""


def to_traditional_chinese(text: str) -> str:
    """Convert Chinese characters without changing Latin text or punctuation."""
    if not isinstance(text, str) or not text.strip():
        raise TraditionalChineseError("沒有可轉換的中文文字")
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            mapper = ctypes.WinDLL("kernel32", use_last_error=True).LCMapStringEx
            mapper.argtypes = [
                wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPCWSTR, ctypes.c_int,
                wintypes.LPWSTR, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_ssize_t,
            ]
            mapper.restype = ctypes.c_int
            # -1 counts UTF-16 code units and the terminator, including emoji.
            flags = 0x04000000  # LCMAP_TRADITIONAL_CHINESE
            size = mapper("zh-TW", flags, text, -1, None, 0, None, None, 0)
            if not size:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_unicode_buffer(size)
            if not mapper("zh-TW", flags, text, -1, buffer, size, None, None, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            converted = buffer.value
        elif sys.platform == "darwin":
            from Foundation import NSString

            source = NSString.stringWithString_(text)
            converted = source.stringByApplyingTransform_reverse_("Hans-Hant", False)
        else:
            raise TraditionalChineseError("這個作業系統尚未提供本機繁體轉換")
    except TraditionalChineseError:
        raise
    except Exception as error:
        raise TraditionalChineseError("本機繁體轉換無法使用，請確認作業系統的中文支援") from error
    if not converted:
        raise TraditionalChineseError("本機繁體轉換沒有產生結果")
    return str(converted)
