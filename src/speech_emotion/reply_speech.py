"""Generate local reply audio with the installed Windows Chinese voices."""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


logger = logging.getLogger(__name__)


class ReplySpeechError(ValueError):
    """Raised when a reply cannot be synthesized locally."""


_SYNTHESIZE_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
$request = ConvertFrom-Json ([Console]::In.ReadToEnd())
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime]
[void][Windows.Media.SpeechSynthesis.SpeechSynthesisStream, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime]
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$stream = $null
$reader = $null
$file = $null
try {
    $voices = @([Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -eq 'zh-TW' })
    if (!$voices.Count) { throw 'No installed Traditional Chinese voice.' }
    $gender = if ($request.mode -eq 'gentle') { 'Female' } else { 'Male' }
    $voice = $voices | Where-Object { $_.Gender.ToString() -eq $gender } | Select-Object -First 1
    if (!$voice) { $voice = $voices[0] }
    $synth.Voice = $voice
    $synth.Options.SpeakingRate = if ($request.mode -eq 'gentle') { 0.94 } else { 1.0 }
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
        $_.GetParameters().Count -eq 1 -and $_.GetGenericArguments().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    $operation = $synth.SynthesizeTextToStreamAsync($request.text)
    $task = $asTask.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]).Invoke($null, @($operation))
    if (!$task.Wait(90000)) { throw 'Speech synthesis timed out.' }
    $stream = $task.Result
    $reader = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream.GetInputStreamAt(0))
    $file = [System.IO.File]::Open($request.output_path, [System.IO.FileMode]::Create)
    $reader.CopyTo($file)
} finally {
    if ($file) { $file.Dispose() }
    if ($reader) { $reader.Dispose() }
    if ($stream) { $stream.Dispose() }
    $synth.Dispose()
}
"""


def synthesize_reply(text: str, mode: str, output_directory: Path) -> Path:
    """Cache a WAV for one reply without passing reply text through the shell."""
    text = text.strip()
    if not text:
        raise ReplySpeechError("尚無可播放的回覆，請先傳送一則訊息。")
    if mode not in {"gentle", "neutral"}:
        raise ReplySpeechError("請選擇可用的回覆聲線。")
    output_directory.mkdir(parents=True, exist_ok=True)
    adapter = "windows-speech-v1" if os.name == "nt" else "macos-say-v1"
    digest = hashlib.sha256(f"{adapter}:{mode}:{text}".encode("utf-8")).hexdigest()
    destination = output_directory / f"{digest}.wav"
    if destination.is_file() and destination.stat().st_size > 44:
        return destination

    if sys.platform == "darwin":
        return _synthesize_macos(text, mode, destination)
    if os.name != "nt":
        raise ReplySpeechError("目前的本機語音輸出只支援 macOS 或 Windows 中文語音。")

    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    script = base64.b64encode(_SYNTHESIZE_SCRIPT.encode("utf-16le")).decode("ascii")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=output_directory, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        payload = {"text": text, "mode": mode, "output_path": str(temporary_path.resolve())}
        subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-EncodedCommand", script],
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if temporary_path.stat().st_size <= 44:
            raise ReplySpeechError("語音合成未產生音訊，請再試一次。")
        temporary_path.replace(destination)
    except (OSError, subprocess.SubprocessError) as error:
        details = error.stderr.decode("utf-8", errors="replace") if isinstance(error, subprocess.CalledProcessError) else str(error)
        logger.error("Windows reply speech failed: %s", details)
        raise ReplySpeechError("無法產生回覆語音，請確認 Windows 已安裝繁體中文語音。") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _synthesize_macos(text: str, mode: str, destination: Path) -> Path:
    """Generate Taiwanese Mandarin locally without sending text to a cloud service."""
    say = shutil.which("say") or "/usr/bin/say"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError) as error:
            raise ReplySpeechError("找不到 FFmpeg，無法轉換 macOS 回覆語音。") from error

    voice = "Meijia" if mode == "gentle" else "Reed"
    rate = "175" if mode == "gentle" else "190"
    aiff_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", dir=destination.parent, delete=False) as temporary:
            aiff_path = Path(temporary.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=destination.parent, delete=False) as temporary:
            wav_path = Path(temporary.name)
        subprocess.run(
            [say, "-v", voice, "-r", rate, "-o", str(aiff_path)],
            input=text, text=True, capture_output=True, check=True, timeout=120,
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(aiff_path), "-ac", "1", "-ar", "24000", str(wav_path)],
            capture_output=True, check=True, timeout=120,
        )
        if wav_path.stat().st_size <= 44:
            raise ReplySpeechError("語音合成未產生音訊，請再試一次。")
        wav_path.replace(destination)
    except (OSError, subprocess.SubprocessError) as error:
        logger.error("macOS reply speech failed: %s", error)
        raise ReplySpeechError("無法產生回覆語音，請確認 macOS 中文語音可以使用。") from error
    finally:
        if aiff_path is not None:
            aiff_path.unlink(missing_ok=True)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    return destination
