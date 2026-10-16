"""Traditional-Chinese Gradio UI for the local personal emotion demo."""

from pathlib import Path
import json

# This application only renders text and audio controls.  Prevent Gradio's
# optional image-component import from spending minutes scanning Pillow image
# plugins at every cold start on the demo Mac.
from PIL import Image

Image._initialized = 2

import gradio as gr

from speech_emotion.personal_demo import (
    PERSONAL_LABELS,
    PersonalDemoError,
    add_manual_training_audio,
    correct_analysis_and_retrain,
    predict_personal_demo,
    train_personal_demo,
    training_counts,
    undo_last_correction_and_retrain,
    undo_last_manual_training_audio,
)
from speech_emotion.audio import denoise_and_normalize_audio
from speech_emotion.integrated_demo import AnalysisServices, CombinedAnalysis, analyse_recording, correct_combined_analysis
from speech_emotion.ollama_client import OllamaClient, OllamaUnavailableError
from speech_emotion.speaker_identity import (
    IdentityResult,
    IDENTITY_THRESHOLD,
    IDENTITY_MARGIN,
    KNOWN_SPEAKERS,
    UNKNOWN_SPEAKER,
    SpeechBrainEmbedder,
    enroll_speaker,
    identify_speaker,
    rebuild_speaker_profiles,
)
from speech_emotion.transcription import (
    OpenAIWhisperTranscriber,
    save_transcript_correction,
    TranscriptResult,
)
from speech_emotion.workspace import DEFAULT_WORKSPACE_ROOT, ProjectWorkspace


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_ROOT = REPOSITORY_ROOT / "data" / "raw" / "personal_demo"
MODEL_PATH = REPOSITORY_ROOT / "models" / "personal_demo.joblib"
WORKSPACE = ProjectWorkspace(DEFAULT_WORKSPACE_ROOT)


class _LocalIdentity:
    def __init__(self) -> None:
        self.embedder = SpeechBrainEmbedder()

    def identify(self, path: Path):
        return identify_speaker(
            path, WORKSPACE, self.embedder, threshold=IDENTITY_THRESHOLD, margin=IDENTITY_MARGIN,
        )


class _LocalEmotion:
    def predict(self, path: Path) -> dict[str, object]:
        return predict_personal_demo(path, MODEL_PATH)


_IDENTITY = _LocalIdentity()
_SERVICES = AnalysisServices(
    identity=_IDENTITY,
    emotion=_LocalEmotion(),
    transcriber=OpenAIWhisperTranscriber(),
)

LABEL_TEXT = {
    "positive": "正面",
    "negative": "負面",
    "calm": "平靜",
}

SUPPORTED_AUDIO_UPLOAD_TYPES = [
    "audio",
    ".webm",
    ".m4a",
    ".aac",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".wma",
]


APPLE_STYLE_CSS = """
:root { color-scheme: light dark; --vs-page: #f5f5f7; --vs-surface: rgba(255,255,255,.80); --vs-text: #1d1d1f; --vs-secondary: #6e6e73; --vs-line: rgba(60,60,67,.18); --vs-blue: #0071e3; --vs-red: #d70015; --vs-result: rgba(0,113,227,.055); --vs-result-line: rgba(0,113,227,.18); }
:host, html, body { width: 100% !important; min-width: 100% !important; min-height: 100% !important; margin: 0 !important; padding: 0 !important; background: var(--vs-page) !important; }
:host { display: block !important; min-height: 100vh !important; }
#root, gradio-app, .gradio-container, .gradio-container > .main, .gradio-container > .main > .wrap { display: block !important; width: 100% !important; max-width: none !important; min-width: 100% !important; min-height: 100vh !important; margin: 0 !important; background: var(--vs-page) !important; color: var(--vs-text) !important; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang TC", "Noto Sans TC", sans-serif !important; }
.gradio-container { padding: 36px clamp(24px, 5vw, 72px) 56px !important; box-sizing: border-box !important; overflow-x: hidden !important; }
.gradio-container > .main, .gradio-container > .main > .wrap { padding: 0 !important; box-sizing: border-box !important; }
#hero { width: min(100%, 1280px) !important; box-sizing: border-box; margin: 0 auto 24px !important; padding: 34px 36px 30px; color: #fff; border: 1px solid rgba(255,255,255,.24); border-radius: 26px; background: radial-gradient(circle at 82% 0%, rgba(127,214,255,.62), transparent 36%), linear-gradient(135deg, #102a43 0%, #0066cc 53%, #5ac8fa 120%); box-shadow: 0 18px 44px rgba(0,74,153,.17); }
#hero h1 { margin: 0 0 8px; color: #fff; font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 700; letter-spacing: -.055em; line-height: 1.08; }
#hero p { max-width: 640px; margin: 0; color: rgba(255,255,255,.84); font-size: 1.04rem; line-height: 1.55; }
#hero .eyebrow { margin-bottom: 12px; color: rgba(255,255,255,.78); font-size: .77rem; font-weight: 650; letter-spacing: .08em; }
.section-card, .section-card > .wrap, .section-card > .wrap > .form, .section-card > .form { width: min(100%, 1280px) !important; margin: 16px auto !important; padding: 26px 0 !important; overflow: visible; border: 0 !important; border-radius: 0 !important; background: transparent !important; box-shadow: none !important; }
.section-heading { margin: 0 0 20px; }.section-heading .step { display: inline-block; margin-bottom: 6px; color: var(--vs-blue); font-size: .78rem; font-weight: 700; letter-spacing: .06em; }.section-heading h2 { margin: 0; color: var(--vs-text); font-size: 1.38rem; font-weight: 650; letter-spacing: -.025em; }.section-heading p { margin: 7px 0 0; color: var(--vs-secondary); font-size: .94rem; line-height: 1.45; }
.section-card .prose, .section-card .prose *, .section-card .markdown, .section-card .markdown * { color: var(--vs-text) !important; }.section-card .section-heading .step { color: var(--vs-blue) !important; }.section-card .section-heading p { color: var(--vs-secondary) !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .wrap { border-radius: 12px; }.gradio-container label span, .gradio-container .label-wrap span { color: var(--vs-text) !important; font-weight: 600 !important; }.gradio-container input, .gradio-container textarea, .gradio-container .wrap-inner, .gradio-container .file-preview, .gradio-container .audio-container { border-color: var(--vs-line) !important; }
.gradio-container button { min-height: 42px; border-radius: 999px !important; font-weight: 600 !important; transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important; }.gradio-container button:hover { transform: translateY(-1px); }.gradio-container button.primary { border: 0 !important; background: var(--vs-blue) !important; box-shadow: 0 5px 14px rgba(0,113,227,.22) !important; }.gradio-container button.primary:hover { background: #0077ed !important; }
#undo-manual, #undo-correction { color: var(--vs-red) !important; border-color: rgba(215,0,21,.34) !important; background: rgba(215,0,21,.04) !important; }.result-panel { min-height: 92px; padding: 16px 18px !important; border: 1px solid var(--vs-result-line) !important; border-radius: 14px !important; background: var(--vs-result) !important; }.result-panel h3 { margin-top: 0 !important; color: var(--vs-blue) !important; }
.correction-area { margin-top: 22px; padding-top: 22px; border-top: 1px solid var(--vs-line); }.correction-area h3 { margin: 0 0 5px; color: var(--vs-text); font-size: 1rem; }.correction-area p { margin: 0 0 14px; color: var(--vs-secondary); font-size: .9rem; }
.training-input-row { align-items: stretch !important; }.training-input-column { min-width: 0 !important; }
@media (prefers-color-scheme: dark) { :root { --vs-page: #1c1c1e; --vs-surface: rgba(28,28,30,.82); --vs-text: #f5f5f7; --vs-secondary: #aeaeb2; --vs-line: rgba(235,235,245,.18); --vs-blue: #0a84ff; --vs-red: #ff453a; --vs-result: rgba(10,132,255,.13); --vs-result-line: rgba(10,132,255,.32); } :host, html, body, #root, gradio-app, .gradio-container, .gradio-container > .main, .gradio-container > .main > .wrap { background: var(--vs-page) !important; } #undo-manual, #undo-correction { border-color: rgba(255,69,58,.48) !important; background: rgba(255,69,58,.12) !important; } }
@media (max-width: 720px) { .gradio-container { padding: 20px 14px 36px !important; } #hero { padding: 28px 24px; border-radius: 20px; } .section-card { padding: 20px !important; border-radius: 16px !important; } }
"""

APPLE_THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Noto Sans TC"), "Arial", "sans-serif"],
)


def _label_text(label: object) -> str:
    """Return the Chinese display label without exposing internal values."""
    return LABEL_TEXT.get(str(label), "未知")


def _counts_text(counts: dict[str, int]) -> str:
    """Format the current recording count for every required class."""
    return "\n".join(
        f"- {_label_text(label)}：{counts.get(label, 0)} 段"
        for label in PERSONAL_LABELS
    )


def format_analysis(result: dict[str, object]) -> str:
    """Render a prediction result as a clear Traditional-Chinese summary."""
    if result.get("status") != "ok":
        return "### 分析失敗\n\n請確認已完成訓練並提供可用的錄音檔後再試。"

    scores = result.get("scores")
    if not isinstance(scores, dict):
        return "### 分析失敗\n\n分析結果不完整，請重新錄音後再試。"

    try:
        latency_ms = float(result["latency_ms"])
    except (KeyError, TypeError, ValueError):
        return "### 分析失敗\n\n分析結果不完整，請重新錄音後再試。"

    score_lines = []
    for label in PERSONAL_LABELS:
        try:
            percentage = float(scores.get(label, 0.0)) * 100
        except (TypeError, ValueError):
            percentage = 0.0
        score_lines.append(f"- {_label_text(label)}：{percentage:.1f}%")

    return "\n".join(
        [
            "### 分析結果",
            "",
            f"最終判斷：{_label_text(result.get('emotion'))}",
            "",
            "信心分數：",
            *score_lines,
            "",
            f"分析時間：{latency_ms:.1f} ms",
        ]
    )


def format_training(result: dict[str, object]) -> str:
    """Render within-speaker evaluation metrics in Traditional Chinese."""
    if result.get("status") == "error":
        return "### 訓練失敗\n\n請確認三種情緒各至少有 2 段可用錄音後再試。"

    try:
        accuracy = float(result["same_speaker_accuracy"]) * 100
        macro_f1 = float(result["same_speaker_macro_f1"]) * 100
        train_count = int(result["same_speaker_train_count"])
        test_count = int(result["same_speaker_test_count"])
    except (KeyError, TypeError, ValueError):
        return "### 訓練失敗\n\n訓練結果不完整，請確認錄音後再試。"

    return "\n".join(
        [
            "### 訓練完成",
            "",
            "同一位說話者測試",
            "",
            f"- Accuracy：{accuracy:.1f}%",
            f"- Macro F1：{macro_f1:.1f}%",
            f"- 訓練錄音：{train_count} 段",
            f"- 測試錄音：{test_count} 段",
        ]
    )


def _recording_status(counts: dict[str, int], heading: str) -> str:
    return f"### {heading}\n\n{_counts_text(counts)}"


def select_input_audio(
    microphone_path: str | None, uploaded_path: str | None
) -> str | None:
    """Use an uploaded file when present, otherwise use the microphone clip."""
    return uploaded_path or microphone_path


def select_training_audio_paths(
    microphone_path: str | None, uploaded_paths: str | list[str] | None
) -> list[str]:
    """Keep every uploaded training file, or use one microphone recording."""
    if isinstance(uploaded_paths, list):
        return uploaded_paths
    if uploaded_paths:
        return [uploaded_paths]
    return [microphone_path] if microphone_path else []


def source_kind_for_training_recording(
    microphone_path: str | None, uploaded_paths: str | list[str] | None
) -> str:
    """Map the training control used to one archive source kind."""
    return "upload" if uploaded_paths else "microphone"


def _add_recording(
    label: str, microphone_path: str | None, uploaded_paths: str | list[str] | None
) -> str:
    """Store one clip and always return a display-safe Chinese response."""
    audio_paths = select_training_audio_paths(microphone_path, uploaded_paths)
    if not audio_paths:
        return _recording_status(
            training_counts(RECORDINGS_ROOT), "請先錄音或上傳音訊檔"
        )
    successful_count = 0
    failed_count = 0
    for audio_path in audio_paths:
        try:
            add_manual_training_audio(Path(audio_path), label, RECORDINGS_ROOT)
            successful_count += 1
        except PersonalDemoError:
            failed_count += 1
        except Exception:
            failed_count += 1
    counts = training_counts(RECORDINGS_ROOT)
    if successful_count == 0:
        return _recording_status(
            training_counts(RECORDINGS_ROOT), "無法加入錄音，請重新錄製或上傳可用音訊檔"
        )
    if failed_count:
        return _recording_status(
            counts, f"已加入 {successful_count} 段錄音；{failed_count} 段無法使用"
        )
    return _recording_status(counts, f"已加入 {successful_count} 段訓練錄音")


def _undo_last_manual_recording() -> str:
    """Undo only the newest recording added through the top training section."""
    try:
        counts = undo_last_manual_training_audio(RECORDINGS_ROOT)
    except PersonalDemoError as error:
        return _recording_status(training_counts(RECORDINGS_ROOT), f"無法撤回：{error}")
    except Exception:
        return _recording_status(training_counts(RECORDINGS_ROOT), "無法撤回加入錄音")
    return _recording_status(counts, "已撤回上一筆加入的訓練錄音")


def _train() -> str:
    """Train locally and suppress implementation errors from the UI."""
    try:
        return format_training(train_personal_demo(RECORDINGS_ROOT, MODEL_PATH))
    except PersonalDemoError:
        return format_training({"status": "error"})
    except Exception:
        return "### 訓練失敗\n\n訓練時發生問題，請確認錄音檔後再試。"


def _analyse(microphone_path: str | None, uploaded_path: str | None) -> str:
    """Analyse one clip locally and suppress implementation errors from the UI."""
    audio_path = select_input_audio(microphone_path, uploaded_path)
    if not audio_path:
        return "### 分析失敗\n\n請先錄音或上傳音訊檔後再試。"
    try:
        return format_analysis(predict_personal_demo(Path(audio_path), MODEL_PATH))
    except PersonalDemoError:
        return format_analysis({"status": "error"})
    except Exception:
        return "### 分析失敗\n\n分析時發生問題，請重新錄音後再試。"


def _correct_and_retrain(
    corrected_label: str, microphone_path: str | None, uploaded_path: str | None
) -> str:
    """Save the selected correction and retrain the local personal model."""
    audio_path = select_input_audio(microphone_path, uploaded_path)
    if not audio_path:
        return "### 無法更正\n\n請先在「立即分析」錄音或上傳音訊檔。"
    try:
        result = correct_analysis_and_retrain(
            Path(audio_path), corrected_label, RECORDINGS_ROOT, MODEL_PATH
        )
    except PersonalDemoError:
        return "### 無法更正\n\n錄音無法使用或三種情緒資料不足，舊模型仍可使用。"
    except Exception:
        return "### 無法重新訓練\n\n更正錄音已保留，但重新訓練失敗；舊模型仍可使用。"

    return "\n".join(
        [
            "### 已更正並重新訓練",
            "",
            f"已將這段錄音標記為：{_label_text(corrected_label)}。",
            "",
            "同一位說話者測試",
            f"- Accuracy：{float(result['same_speaker_accuracy']) * 100:.1f}%",
            f"- Macro F1：{float(result['same_speaker_macro_f1']) * 100:.1f}%",
        ]
    )


def _undo_last_correction() -> str:
    """Undo only the newest correction and retrain the local personal model."""
    try:
        result = undo_last_correction_and_retrain(RECORDINGS_ROOT, MODEL_PATH)
    except PersonalDemoError as error:
        return f"### 無法撤回\n\n{error}"
    except Exception:
        return "### 無法撤回\n\n重新訓練失敗，原本更正已保留，舊模型仍可使用。"

    return "\n".join(
        [
            "### 已撤回上一筆更正並重新訓練",
            "",
            "同一位說話者測試",
            f"- Accuracy：{float(result['same_speaker_accuracy']) * 100:.1f}%",
            f"- Macro F1：{float(result['same_speaker_macro_f1']) * 100:.1f}%",
        ]
    )


def _format_combined_analysis(result: object) -> tuple[str, str, str, str, str]:
    """Render one archived combined analysis and retain only its safe record id."""
    if getattr(result, "status", "error") != "ok":
        message = getattr(result, "message", None) or "請確認錄音、模型與本機套件後再試。"
        return ("### 分析失敗\n\n" + str(message), "尚無結果", "尚無結果", "", "")
    identity = result.identity
    emotion = result.emotion or {}
    transcript = result.transcript
    speaker_lines = [f"### 說話者：{identity.speaker}", "", "相似度："]
    speaker_lines += [f"- {name}：{score * 100:.1f}%" for name, score in identity.scores.items()]
    emotion_lines = [f"### 情緒：{_label_text(emotion.get('emotion'))}", "", "信心分數："]
    emotion_lines += [f"- {_label_text(label)}：{float(emotion.get('scores', {}).get(label, 0)) * 100:.1f}%" for label in PERSONAL_LABELS]
    transcript_text = transcript.effective_text if transcript else ""
    summary = "\n".join([
        "### 已完成本機分析", "",
        f"紀錄編號：`{result.record_id}`", "",
        f"總分析時間：{result.total_latency_ms:.0f} ms", "",
        "原始檔、降噪 WAV 與結果已自動保存到桌面「專題研究」。",
    ])
    return (summary, "\n".join(speaker_lines), "\n".join(emotion_lines), transcript_text, result.record_id)


def _analyse_integrated(microphone_path: str | None, uploaded_path: str | None):
    audio_path = select_input_audio(microphone_path, uploaded_path)
    if not audio_path:
        return ("### 請先錄音或上傳音訊檔。", "尚無結果", "尚無結果", "", "")
    try:
        return _format_combined_analysis(analyse_recording(Path(audio_path), "upload" if uploaded_path else "microphone", WORKSPACE, _SERVICES))
    except Exception as error:
        return (f"### 分析失敗\n\n{error}", "尚無結果", "尚無結果", "", "")


def _enroll_speaker_recordings(speaker: str, microphone_path: str | None, uploaded_paths: str | list[str] | None) -> str:
    paths = select_training_audio_paths(microphone_path, uploaded_paths)
    if not paths:
        return "### 請先錄音或上傳至少一段成員語音。"
    succeeded = 0
    failed = 0
    added_paths: list[str] = []
    source_kind = source_kind_for_training_recording(microphone_path, uploaded_paths)
    for path in paths:
        try:
            archived = WORKSPACE.archive_input(Path(path), source_kind)
            target = WORKSPACE.root / "03_語音身分辨識" / "處理後音檔" / f"{archived.record_id}.wav"
            processed = denoise_and_normalize_audio(archived.original_path, target)
            enrolled_path = enroll_speaker(processed, speaker, WORKSPACE)
            added_paths.append(str(enrolled_path))
            succeeded += 1
        except Exception:
            failed += 1
    if not succeeded:
        return "### 無法加入聲紋錄音\n\n請確認音檔長度至少一秒，而且不是靜音。"
    history_path = WORKSPACE.root / "03_語音身分辨識" / "enrollment_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if not isinstance(history, list):
            history = []
    except (OSError, json.JSONDecodeError):
        history = []
    history.append({"speaker": speaker, "paths": added_paths})
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return f"### 已保存 {succeeded} 段 {speaker} 的聲紋錄音" + (f"\n\n另有 {failed} 段無法使用。" if failed else "\n\n請按「建立／更新聲紋範本」。")


def _undo_last_speaker_recording() -> str:
    """Remove the newest speaker-enrollment batch and rebuild its profiles."""
    history_path = WORKSPACE.root / "03_語音身分辨識" / "enrollment_history.json"
    if not history_path.exists():
        return "### 無法刪除\n\n目前沒有可刪除的聲紋錄音。"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "### 無法刪除\n\n聲紋加入紀錄無法讀取。"
    if not isinstance(history, list) or not history:
        return "### 無法刪除\n\n目前沒有可刪除的聲紋錄音。"
    entry = history[-1]
    if not isinstance(entry, dict) or not isinstance(entry.get("paths"), list):
        return "### 無法刪除\n\n聲紋加入紀錄格式不正確。"
    removed = 0
    allowed_root = (WORKSPACE.root / "01_原始音檔" / "身分辨識").resolve()
    for raw_path in entry["paths"]:
        if not isinstance(raw_path, str):
            continue
        target = Path(raw_path).resolve()
        try:
            target.relative_to(allowed_root)
        except ValueError:
            continue
        if target.suffix.lower() == ".wav" and target.is_file():
            target.unlink()
            removed += 1
            (WORKSPACE.root / "03_語音身分辨識" / "聲紋向量" / f"{target.stem}.npy").unlink(missing_ok=True)
    history_path.write_text(json.dumps(history[:-1], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        counts = rebuild_speaker_profiles(WORKSPACE, _IDENTITY.embedder)
    except Exception as error:
        return f"### 已刪除 {removed} 段聲紋錄音，但更新範本失敗\n\n{error}"
    lines = ["### 已刪除上一筆聲紋錄音並更新範本", ""]
    lines += [f"- {name}：{counts.get(name, 0)} 段" for name in KNOWN_SPEAKERS]
    return "\n".join(lines)


def _rebuild_speakers() -> str:
    try:
        counts = rebuild_speaker_profiles(WORKSPACE, _IDENTITY.embedder)
    except Exception as error:
        return f"### 建立聲紋範本失敗\n\n{error}"
    lines = ["### 聲紋範本已更新", ""]
    lines += [f"- {speaker}：{counts.get(speaker, 0)} 段" for speaker in KNOWN_SPEAKERS]
    return "\n".join(lines)


def _save_combined_correction(record_id: str, speaker: str, emotion: str, transcript: str) -> tuple[str, str, str, str]:
    if not record_id:
        return ("### 請先完成一次分析，再儲存更正。", "", "", "")
    try:
        result = correct_combined_analysis(record_id, speaker, emotion, transcript, WORKSPACE, _SERVICES)
        if transcript.strip():
            save_transcript_correction(record_id, transcript, WORKSPACE)
        retrain_note = ""
        if result.processed_path and speaker in KNOWN_SPEAKERS:
            enroll_speaker(result.processed_path, speaker, WORKSPACE)
            rebuild_speaker_profiles(WORKSPACE, _IDENTITY.embedder)
            retrain_note = "\n\n已用更正說話者更新本機聲紋範本。"
        if result.processed_path and emotion in PERSONAL_LABELS:
            try:
                correct_analysis_and_retrain(result.processed_path, emotion, RECORDINGS_ROOT, MODEL_PATH)
                retrain_note += "\n\n已用更正情緒重新訓練情緒模型。"
            except PersonalDemoError:
                retrain_note += "\n\n情緒更正已保存；目前三類資料不足，暫時未重新訓練。"
        formatted = _format_combined_analysis(result)
        return ("### 更正已保存" + retrain_note, formatted[1], formatted[2], transcript)
    except Exception as error:
        return (f"### 無法儲存更正\n\n{error}", "", "", "")


def _ask_ollama(record_id: str, message: str, model: str) -> str:
    if not record_id or not message.strip():
        return "請先完成一次分析，並輸入想對聊天機器人說的話。"
    try:
        import json
        payload = json.loads((WORKSPACE.root / "05_完整分析紀錄" / "records" / f"{record_id}.json").read_text(encoding="utf-8"))
        identity_data = payload["identity"]
        transcript_data = payload["transcript"]
        analysis = CombinedAnalysis(
            record_id=record_id,
            archive_path=Path(str(payload["archive_path"])),
            processed_path=Path(str(payload["processed_path"])) if payload.get("processed_path") else None,
            identity=IdentityResult(str(identity_data["speaker"]), dict(identity_data["scores"]), bool(identity_data["is_unknown"]), float(identity_data["latency_ms"])),
            emotion=dict(payload.get("emotion") or {}),
            transcript=TranscriptResult(record_id, str(transcript_data["original_text"]), transcript_data.get("corrected_text"), transcript_data.get("cer"), float(transcript_data["latency_ms"]), transcript_data.get("traditional_text")),
            total_latency_ms=float(payload["total_latency_ms"]), status=str(payload["status"]), message=payload.get("message"),
        )
        return OllamaClient().chat(analysis, message.strip(), model.strip())
    except OllamaUnavailableError as error:
        return f"無法連接本機 Ollama：{error}"
    except Exception as error:
        return f"聊天失敗：{error}"


def _text_only_analysis(message: str) -> CombinedAnalysis:
    """Create an explicitly unknown context for a text-only chat turn."""
    return CombinedAnalysis(
        record_id="text-only",
        archive_path=Path(),
        processed_path=None,
        identity=IdentityResult(UNKNOWN_SPEAKER, {name: -1.0 for name in KNOWN_SPEAKERS}, True, 0.0),
        emotion={"emotion": "calm", "scores": {label: 0.0 for label in PERSONAL_LABELS}},
        transcript=TranscriptResult("text-only", message, None, None, 0.0),
        total_latency_ms=0.0,
        status="ok",
    )


def _chat_with_voice(
    message: str,
    microphone_path: str | None,
    uploaded_path: str | None,
    history: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]], str, str, str, str, str, str]:
    """Run local voice analysis when provided, then ask the approved local LLM."""
    history = list(history or [])
    audio_path = select_input_audio(microphone_path, uploaded_path)
    typed_message = message.strip()
    try:
        if audio_path:
            analysis = analyse_recording(
                Path(audio_path), "upload" if uploaded_path else "microphone", WORKSPACE, _SERVICES
            )
            if analysis.status != "ok":
                raise ValueError(analysis.message or "語音分析失敗")
            user_text = analysis.transcript.effective_text if analysis.transcript else typed_message
            if typed_message:
                user_text = typed_message
            formatted = _format_combined_analysis(analysis)
        elif typed_message:
            analysis = _text_only_analysis(typed_message)
            user_text = typed_message
            formatted = ("### 文字聊天模式", "說話者：未知使用者", "情緒：未分析", typed_message, "")
        else:
            return history, "請輸入文字，或錄音／上傳音檔。", "", "", "", "", ""
        reply = OllamaClient().chat(analysis, user_text, "gemma4:12b", history=history)
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ])
        return history, "", formatted[0], formatted[1], formatted[2], formatted[3], analysis.record_id
    except OllamaUnavailableError as error:
        return history, f"本機 Gemma 暫時無法回覆：{error}", "", "", "", "", ""
    except Exception as error:
        return history, f"處理失敗：{error}", "", "", "", "", ""


def _clear_chat() -> tuple[list[dict[str, str]], str, str, str, str, str, str]:
    """Clear only visible chat state; archived training and analysis data remain local."""
    return [], "", "", "", "", "", ""


def build_demo() -> gr.Blocks:
    """Build the local-only personal three-emotion Gradio page."""
    initial_counts = training_counts(RECORDINGS_ROOT)
    with gr.Blocks(title="具身分辨識與情緒感知的聊天機器人｜模型訓練後台") as demo:
        gr.Markdown(
            "<div id='hero'><div class='eyebrow'>LOCAL SPEECH EMOTION DEMO</div>"
            "<h1>模型訓練後台</h1>"
            "<p>用你的聲音建立專屬的三情緒模型，完成訓練後，立即查看每段語音的正面、負面與平靜傾向。</p></div>"
        )

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown(
                "<div class='section-heading'><div class='step'>STEP 01</div>"
                "<h2>加入訓練錄音</h2><p>選擇情緒後錄音，或一次上傳多個常見音訊檔。</p></div>"
            )
            with gr.Row(equal_height=True, elem_classes=["training-input-row"]):
                with gr.Column(scale=1, min_width=0, elem_classes=["training-input-column"]):
                    label = gr.Dropdown(
                        choices=[(LABEL_TEXT[key], key) for key in PERSONAL_LABELS],
                        value="positive",
                        label="情緒標籤",
                    )
                with gr.Column(scale=1, min_width=0, elem_classes=["training-input-column"]):
                    training_microphone = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="使用麥克風錄音",
                    )
                with gr.Column(scale=1, min_width=0, elem_classes=["training-input-column"]):
                    training_upload = gr.File(
                        file_types=SUPPORTED_AUDIO_UPLOAD_TYPES,
                        file_count="multiple",
                        type="filepath",
                        label="或上傳音訊檔（支援 WebM、MP3、M4A 等）",
                    )
            with gr.Row():
                add_button = gr.Button("加入訓練錄音", variant="primary")
                undo_manual_button = gr.Button("撤回上一筆加入的訓練錄音", elem_id="undo-manual")
            add_status = gr.Markdown(_recording_status(initial_counts, "目前訓練錄音"), elem_classes=["result-panel"])

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown(
                "<div class='section-heading'><div class='step'>STEP 02</div>"
                "<h2>訓練個人模型</h2><p>三種情緒各至少需要兩段有效錄音；資料越多，判斷越穩定。</p></div>"
            )
            train_button = gr.Button("訓練個人 Demo", variant="primary")
            training_result = gr.Markdown("尚未訓練。", elem_classes=["result-panel"])

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown(
                "<div class='section-heading'><div class='step'>STEP 03</div>"
                "<h2>立即分析</h2><p>錄製或上傳一段新的聲音，查看模型判斷與三種情緒的信心分數。</p></div>"
            )
            analysis_microphone = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="使用麥克風錄音",
            )
            analysis_upload = gr.File(
                file_types=SUPPORTED_AUDIO_UPLOAD_TYPES,
                type="filepath",
                label="或上傳音訊檔（支援 WebM、MP3、M4A 等）",
            )
            analyse_button = gr.Button("立即分析", variant="primary")
            analysis_result = gr.Markdown("請錄音或上傳音訊檔。", elem_classes=["result-panel"])

            gr.Markdown(
                "<div class='correction-area'><h3>判斷錯誤？更正情緒</h3>"
                "<p>選擇正確標籤後重新訓練；若誤按，也能立即撤回最近一次更正。</p></div>"
            )
            with gr.Row():
                corrected_label = gr.Dropdown(
                    choices=[(LABEL_TEXT[key], key) for key in PERSONAL_LABELS],
                    value="positive",
                    label="正確情緒",
                )
                correct_button = gr.Button("更正並重新訓練")
                undo_button = gr.Button("撤回上一筆更正並重新訓練", elem_id="undo-correction")
            correction_result = gr.Markdown("", elem_classes=["result-panel"])

        add_button.click(
            _add_recording,
            inputs=[label, training_microphone, training_upload],
            outputs=add_status,
        )
        undo_manual_button.click(_undo_last_manual_recording, outputs=add_status)
        train_button.click(_train, outputs=training_result)
        analyse_button.click(
            _analyse,
            inputs=[analysis_microphone, analysis_upload],
            outputs=analysis_result,
        )
        correct_button.click(
            _correct_and_retrain,
            inputs=[corrected_label, analysis_microphone, analysis_upload],
            outputs=correction_result,
        )
        undo_button.click(_undo_last_correction, outputs=correction_result)

        gr.Markdown("<div class='section-heading' style='margin-top:42px'><div class='step'>INTEGRATED VOICE IDENTITY · TRANSCRIPT · CHAT</div><h2>完整語音辨識工作台</h2><p>所有上傳與錄音會先自動保存原始檔，再建立降噪 WAV，並在本機完成身分、情緒與中文轉寫。</p></div>")

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("<div class='section-heading'><div class='step'>STEP 04</div><h2>建立四位成員的聲紋範本</h2><p>每位成員建議錄 5 至 10 段、每段 3 至 10 秒的自然中文語音。若信心不足，系統會顯示「未知使用者」。</p></div>")
            with gr.Row(equal_height=True):
                speaker_select = gr.Dropdown(choices=list(KNOWN_SPEAKERS), value=KNOWN_SPEAKERS[0], label="這段聲音是誰說的")
                speaker_microphone = gr.Audio(sources=["microphone"], type="filepath", label="使用麥克風錄音")
                speaker_upload = gr.File(file_types=SUPPORTED_AUDIO_UPLOAD_TYPES, file_count="multiple", type="filepath", label="或一次上傳多段音檔")
            with gr.Row():
                speaker_add_button = gr.Button("保存聲紋錄音", variant="primary")
                speaker_undo_button = gr.Button("刪除上一筆聲紋錄音", elem_id="undo-speaker")
                speaker_rebuild_button = gr.Button("建立／更新聲紋範本（套用錄音）")
            speaker_status = gr.Markdown("尚未建立聲紋範本。", elem_classes=["result-panel"])

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("<div class='section-heading'><div class='step'>STEP 05</div><h2>整合分析一段語音</h2><p>支援麥克風錄音與 WebM、MP3、M4A、WAV、FLAC 等音檔。第一次中文轉寫會下載本機 Whisper 模型，請耐心等候。</p></div>")
            with gr.Row(equal_height=True):
                integrated_microphone = gr.Audio(sources=["microphone"], type="filepath", label="使用麥克風錄音")
                integrated_upload = gr.File(file_types=SUPPORTED_AUDIO_UPLOAD_TYPES, type="filepath", label="或上傳一段音訊檔")
            integrated_button = gr.Button("開始完整分析", variant="primary")
            integrated_summary = gr.Markdown("請錄音或上傳音訊檔。", elem_classes=["result-panel"])
            record_state = gr.State("")
            with gr.Row(equal_height=True):
                speaker_result = gr.Markdown("尚無結果", elem_classes=["result-panel"])
                emotion_result = gr.Markdown("尚無結果", elem_classes=["result-panel"])
            transcript_original = gr.Textbox(label="原始中文轉寫結果", lines=3, interactive=False)

            gr.Markdown("<div class='correction-area'><h3>更正這次分析</h3><p>正確答案會保存到桌面「專題研究」。說話者更正會更新聲紋範本；情緒更正會加入現有情緒訓練資料。</p></div>")
            with gr.Row():
                speaker_correction = gr.Dropdown(choices=[*KNOWN_SPEAKERS, UNKNOWN_SPEAKER], value=UNKNOWN_SPEAKER, label="正確說話者")
                emotion_correction = gr.Dropdown(choices=[(LABEL_TEXT[key], key) for key in PERSONAL_LABELS], value="calm", label="正確情緒")
            transcript_correction = gr.Textbox(label="正確逐字稿（可留白）", lines=3)
            save_correction_button = gr.Button("儲存更正並更新模型")
            correction_status = gr.Markdown("", elem_classes=["result-panel"])

            gr.Markdown("<div class='correction-area'><h3>本機 Ollama 聊天</h3><p>會把這次的說話者、情緒與轉寫內容傳給你電腦上的 Ollama；沒有啟動 Ollama 時不會送到網路。</p></div>")
            chat_message = gr.Textbox(label="想對聊天機器人說的話", lines=2, placeholder="例如：我今天覺得壓力很大")
            chat_model = gr.Textbox(label="本機 Ollama 模型名稱", value="llama3.2:3b")
            chat_button = gr.Button("用本機 Ollama 回覆")
            chat_result = gr.Markdown("", elem_classes=["result-panel"])

        speaker_add_button.click(_enroll_speaker_recordings, inputs=[speaker_select, speaker_microphone, speaker_upload], outputs=speaker_status)
        speaker_undo_button.click(_undo_last_speaker_recording, outputs=speaker_status)
        speaker_rebuild_button.click(_rebuild_speakers, outputs=speaker_status)
        integrated_button.click(_analyse_integrated, inputs=[integrated_microphone, integrated_upload], outputs=[integrated_summary, speaker_result, emotion_result, transcript_original, record_state])
        save_correction_button.click(_save_combined_correction, inputs=[record_state, speaker_correction, emotion_correction, transcript_correction], outputs=[correction_status, speaker_result, emotion_result, transcript_original])
        chat_button.click(_ask_ollama, inputs=[record_state, chat_message, chat_model], outputs=chat_result)
    return demo


def build_integrated_demo() -> gr.Blocks:
    """Build the chat-first final presentation with training controls retained below."""
    initial_counts = training_counts(RECORDINGS_ROOT)
    with gr.Blocks(title="具身分辨識與情緒感知的聊天機器人") as demo:
        gr.Markdown(
            "<div id='hero'><div class='eyebrow'>LOCAL IDENTITY · EMOTION · CHAT</div>"
            "<h1>具身分辨識與情緒感知的聊天機器人</h1><p>先理解你的聲音，再用本機 Gemma 給出有脈絡的中文回覆。所有資料只保存在這台電腦。</p></div>"
        )
        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("<div class='section-heading'><div class='step'>CHAT</div><h2>開始對話</h2><p>可直接輸入文字，或用錄音／音檔完成身分辨識、情緒辨識和中文轉寫後再聊天。</p></div>")
            chat_history = gr.Chatbot(height=420, label="對話紀錄", placeholder="你好，我是具身分辨識與情緒感知的聊天機器人。")
            history_state = gr.State([])
            analysis_record_state = gr.State("")
            with gr.Row(equal_height=True):
                chat_microphone = gr.Audio(sources=["microphone"], type="filepath", label="錄製語音")
                chat_upload = gr.File(file_types=SUPPORTED_AUDIO_UPLOAD_TYPES, type="filepath", label="或上傳音訊檔")
            chat_input = gr.Textbox(label="輸入訊息", placeholder="例如：我最近報告壓力很大，該怎麼開始？", lines=2)
            with gr.Row():
                chat_send = gr.Button("送出並由本機 Gemma 回覆", variant="primary")
                chat_clear = gr.Button("清除這次對話")
            chat_status = gr.Markdown("Gemma 4 12B 已設定為本機聊天模型。", elem_classes=["result-panel"])

        with gr.Accordion("查看本次語音分析與更正", open=False):
            analysis_summary = gr.Markdown("尚未分析語音。", elem_classes=["result-panel"])
            with gr.Row(equal_height=True):
                chat_speaker = gr.Markdown("", elem_classes=["result-panel"])
                chat_emotion = gr.Markdown("", elem_classes=["result-panel"])
            chat_transcript = gr.Textbox(label="本次原始中文轉寫", lines=3, interactive=False)
            with gr.Row():
                chat_speaker_correction = gr.Dropdown(choices=[*KNOWN_SPEAKERS, UNKNOWN_SPEAKER], value=UNKNOWN_SPEAKER, label="正確說話者")
                chat_emotion_correction = gr.Dropdown(choices=[(LABEL_TEXT[key], key) for key in PERSONAL_LABELS], value="calm", label="正確情緒")
            chat_transcript_correction = gr.Textbox(label="正確逐字稿（可留白）", lines=2)
            chat_correction_button = gr.Button("儲存更正並更新模型")
            chat_correction_status = gr.Markdown("", elem_classes=["result-panel"])

        with gr.Accordion("訓練情緒模型", open=False):
            gr.Markdown("三種情緒各至少加入兩段有效錄音，資料越多，三類判斷越穩定。")
            with gr.Row(equal_height=True):
                emotion_label = gr.Dropdown(choices=[(LABEL_TEXT[key], key) for key in PERSONAL_LABELS], value="positive", label="情緒標籤")
                emotion_microphone = gr.Audio(sources=["microphone"], type="filepath", label="錄製訓練語音")
                emotion_upload = gr.File(file_types=SUPPORTED_AUDIO_UPLOAD_TYPES, file_count="multiple", type="filepath", label="一次上傳多段音檔")
            with gr.Row():
                emotion_add = gr.Button("加入情緒訓練資料", variant="primary")
                emotion_undo = gr.Button("撤回最近一筆訓練錄音", elem_id="undo-manual")
                emotion_train = gr.Button("訓練情緒模型")
            emotion_status = gr.Markdown(_recording_status(initial_counts, "目前情緒訓練錄音"), elem_classes=["result-panel"])
            emotion_metrics = gr.Markdown("尚未訓練。", elem_classes=["result-panel"])

        with gr.Accordion("訓練四位成員的聲紋模型", open=False):
            gr.Markdown("每位成員建議保存 5 至 10 段、每段 3 至 10 秒自然中文語音。系統會把低信心結果標為未知使用者。")
            with gr.Row(equal_height=True):
                speaker_select = gr.Dropdown(choices=list(KNOWN_SPEAKERS), value=KNOWN_SPEAKERS[0], label="說話者")
                speaker_microphone = gr.Audio(sources=["microphone"], type="filepath", label="錄製成員語音")
                speaker_upload = gr.File(file_types=SUPPORTED_AUDIO_UPLOAD_TYPES, file_count="multiple", type="filepath", label="一次上傳多段音檔")
            with gr.Row():
                speaker_add = gr.Button("保存聲紋錄音", variant="primary")
                speaker_undo = gr.Button("刪除上一筆聲紋錄音", elem_id="undo-speaker")
                speaker_rebuild = gr.Button("建立／更新聲紋範本（套用錄音）")
            speaker_status = gr.Markdown("尚未建立聲紋範本。", elem_classes=["result-panel"])

        chat_send.click(
            _chat_with_voice,
            inputs=[chat_input, chat_microphone, chat_upload, history_state],
            outputs=[chat_history, chat_status, analysis_summary, chat_speaker, chat_emotion, chat_transcript, analysis_record_state],
        ).then(lambda history: history, inputs=chat_history, outputs=history_state)
        chat_clear.click(_clear_chat, outputs=[chat_history, chat_status, analysis_summary, chat_speaker, chat_emotion, chat_transcript, analysis_record_state]).then(lambda: [], outputs=history_state)
        chat_correction_button.click(
            _save_combined_correction,
            inputs=[analysis_record_state, chat_speaker_correction, chat_emotion_correction, chat_transcript_correction],
            outputs=[chat_correction_status, chat_speaker, chat_emotion, chat_transcript],
        )
        emotion_add.click(_add_recording, inputs=[emotion_label, emotion_microphone, emotion_upload], outputs=emotion_status)
        emotion_undo.click(_undo_last_manual_recording, outputs=emotion_status)
        emotion_train.click(_train, outputs=emotion_metrics)
        speaker_add.click(_enroll_speaker_recordings, inputs=[speaker_select, speaker_microphone, speaker_upload], outputs=speaker_status)
        speaker_undo.click(_undo_last_speaker_recording, outputs=speaker_status)
        speaker_rebuild.click(_rebuild_speakers, outputs=speaker_status)
    return demo
