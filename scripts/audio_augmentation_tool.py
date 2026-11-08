"""Independent local UI; does not load or modify recognition models."""

import argparse
from pathlib import Path
import shutil
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from speech_emotion.reference_augmentation import METHODS, export_reference_batch

OUTPUT_ROOT = ROOT / "outputs" / "reference-augmentation"


def run_export(uploads, recording, methods, rotation, gain, rate, pitch):
    paths = list(uploads or [])
    if recording:
        paths.append(recording)
    try:
        result = export_reference_batch(paths, OUTPUT_ROOT, methods, rotation, gain, rate, pitch)
        # Serve only explicitly selected results from Gradio's permitted temp area.
        download = Path(tempfile.mkdtemp(prefix="audio-experiment-"))
        for path in (result.archive, result.original_preview, *result.outputs):
            shutil.copy2(path, download / path.name)
        variants = {f"{i + 1}. {row[0]} · {row[1]}": str(download / path.name)
                    for i, (path, row) in enumerate(zip(result.outputs, result.rows))}
        first = next(iter(variants))
        return (str(download / result.archive.name), result.rows,
                (download / result.original_preview.name).read_bytes(),
                gr.update(choices=list(variants), value=first), variants,
                Path(variants[first]).read_bytes(), f"已完成：{len(result.outputs)} 份副本；未加入資料庫或訓練。")
    except (ValueError, OSError) as error:
        return (None, [], None, gr.update(choices=[], value=None), {}, None,
                f"無法產生：{error}")


CSS = """
.gradio-container { width: 100% !important; max-width: 1120px !important; min-width: 0 !important; margin: auto; }
.gradio-container .column, .gradio-container .row { min-width: 0 !important; }
@media (max-width: 600px) {
  #experiment-layout { flex-direction: column !important; }
  #experiment-layout > .column { width: 100% !important; flex: none !important; }
}
h1 { font-size: 26px !important; letter-spacing: 0 !important; }
footer { display: none !important; }
button { border-radius: 6px !important; }
"""


def build_app():
    with gr.Blocks(title="聲音擴增實驗室") as app:
        gr.Markdown("# 聲音擴增實驗室")
        with gr.Row(elem_id="experiment-layout"):
            with gr.Column(scale=1, min_width=280):
                uploads = gr.File(label="來源音檔", file_count="multiple", type="filepath",
                                  file_types=["audio"], elem_id="source-files")
                recording = gr.Audio(label="現場錄音", sources=["microphone"], type="filepath")
                methods = gr.CheckboxGroup(choices=[(v, k) for k, v in METHODS.items()],
                                          value=["rotation"], label="擴增方法")
                rotation = gr.Slider(1, 99, value=40, step=1, label="時間旋轉切點（%）")
                gain = gr.Slider(0.25, 1.5, value=0.75, step=0.05, label="音量倍率")
                rate = gr.Slider(0.5, 1.5, value=0.75, step=0.05, label="播放速率倍率（連動音高與時長）")
                pitch = gr.Slider(-6, 6, value=-4, step=0.1, label="固定時長音高（半音）")
                button = gr.Button("產生副本", variant="primary")
                status = gr.Textbox(label="處理狀態", interactive=False)
            with gr.Column(scale=2, min_width=280):
                original = gr.Audio(label="第一個來源音檔", interactive=False)
                selection = gr.Dropdown(label="試聽副本", choices=[], interactive=True)
                processed = gr.Audio(label="處理後音檔", interactive=False)
                table = gr.Dataframe(headers=["來源", "方法", "參數", "原長（秒）", "輸出（秒）"],
                                     column_widths=[160, 110, 70, 100, 100],
                                     interactive=False, wrap=True, label="本批次結果")
                archive = gr.File(label="下載 WAV 與來源清單", interactive=False)
        state = gr.State({})
        button.click(run_export, [uploads, recording, methods, rotation, gain, rate, pitch],
                     [archive, table, original, selection, state, processed, status],
                     api_name="export_audio_experiments", concurrency_limit=1)
        selection.change(lambda choice, variants: Path(variants[choice]).read_bytes()
                         if choice in variants else None, [selection, state], processed)
    return app


def available_port(start):
    for port in range(start, min(start + 20, 65536)):
        try:
            with socket.socket() as connection:
                connection.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("找不到可用連接埠。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7876)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--strict-port", action="store_true")
    args = parser.parse_args()
    port = args.port if args.strict_port else available_port(args.port)
    build_app().queue().launch(server_name="127.0.0.1", server_port=port,
                               inbrowser=not args.no_browser, css=CSS,
                               theme=gr.themes.Default(primary_hue="teal"))
