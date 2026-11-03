"""Local, single-job adapter for the existing EchoMimic CLI."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
PROCESS = None
CANCEL = threading.Event()
PRESETS = {'快速檢查 · 2 步（畫質較低）': 2, '效果測試 · 10 步': 10, '細節測試 · 20 步（較慢）': 20}

def check_duration(value):
    if not math.isfinite(value) or not .5 <= value <= 5:
        raise ValueError('請使用 0.5～5 秒的音訊；本次不會自動裁切。')
    return value

def new_job_dir(base):
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path

def reserve_job():
    if not LOCK.acquire(blocking=False):
        raise ValueError('已有生成工作，請等候完成或按「取消目前工作」。')
    CANCEL.clear()

def release_job():
    LOCK.release()

def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('無法讀取音訊或影片，請換一個有效檔案。')
    return json.loads(result.stdout)

def cancel_job():
    CANCEL.set()
    with STATE_LOCK:
        proc = PROCESS
        if proc is not None and proc.poll() is None:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    return '已送出取消要求，正在停止目前工作。'

def prepare_inputs(image_path, audio_path, job):
    if not image_path or not audio_path:
        raise ValueError('請先選擇一張照片與一段音訊。')
    if Path(image_path).stat().st_size > 20 * 1024 * 1024 or Path(audio_path).stat().st_size > 30 * 1024 * 1024:
        raise ValueError('照片上限 20 MB，音訊上限 30 MB。')
    metadata = probe(audio_path)
    if not any(s['codec_type'] == 'audio' for s in metadata['streams']):
        raise ValueError('檔案沒有音訊軌。')
    duration = check_duration(float(metadata['format']['duration']))
    with Image.open(image_path) as img:
        if img.width * img.height > 24_000_000:
            raise ValueError('照片請小於 2400 萬像素。')
        img = ImageOps.exif_transpose(img).convert('RGB')
        img.thumbnail((1536, 1536))
        img.save(job / 'portrait.png')
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(audio_path), '-vn', '-ac', '1', '-ar', '16000', str(job / 'speech.wav')], check=True, timeout=30, capture_output=True)
    # Use decoded audio duration, not potentially rounded container metadata.
    duration = check_duration(float(probe(job / 'speech.wav')['format']['duration']))
    config = yaml.safe_load((ROOT / 'configs/prompts/animation_mac.yaml').read_text())
    config['test_cases'] = {str(job / 'portrait.png'): [str(job / 'speech.wav')]}
    (job / 'config.yaml').write_text(yaml.safe_dump(config))
    return duration

def result_video(job, duration):
    matches = list((job / 'render').glob('*_withaudio.mp4'))
    if len(matches) != 1:
        raise RuntimeError('本次工作沒有產生完整影片，請查看日誌。')
    info = probe(matches[0])
    streams = {s['codec_type']: s for s in info['streams']}
    if 'video' not in streams or 'audio' not in streams:
        raise RuntimeError('輸出缺少畫面或音軌。')
    vd = float(streams['video']['duration']); ad = float(streams['audio']['duration'])
    if abs(vd - duration) > .15 or abs(ad - duration) > .15:
        raise RuntimeError('輸出長度與輸入音訊不一致，請查看日誌。')
    return str(matches[0])

def run_job(image_path, audio_path, quality):
    global PROCESS
    reserve_job()
    started = time.monotonic()
    proc = None
    job = None
    try:
        if quality not in PRESETS:
            raise ValueError('請選擇測試品質。')
        job = new_job_dir(ROOT / 'output/web_tests')
        yield '正在檢查照片與音訊…', None, None, None
        duration = prepare_inputs(image_path, audio_path, job)
        if not (ROOT / 'pretrained_weights/denoising_unet.pth').is_file():
            raise ValueError('找不到模型權重，請確認 KATHY 外接硬碟已連接。')
        frames = math.ceil(duration * 12)
        command = [sys.executable, '-u', str(ROOT / 'infer_audio2vid.py'), '--config', str(job / 'config.yaml'), '--device', 'mps', '-W', '256', '-H', '256', '-L', str(frames), '--fps', '12', '--steps', str(PRESETS[quality]), '--output-dir', str(job / 'render')]
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1', TOKENIZERS_PARALLELISM='false')
        log_path = job / 'run.log'
        with log_path.open('w') as log:
            with STATE_LOCK:
                if CANCEL.is_set():
                    yield '已取消。', None, None, None
                    return
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                PROCESS = proc
            while proc.poll() is None:
                if CANCEL.is_set():
                    try: os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    break
                if time.monotonic() - started > 3600:
                    cancel_job()
                    raise RuntimeError('工作超過一小時，已停止。')
                text = log_path.read_text(errors='replace')
                phases = [('Loading VAE', '載入影像解碼模型'), ('Loading reference model', '載入人像模型'), ('Loading video model', '載入影片與動作模型'), ('Loading face locator', '載入人臉定位'), ('Loading audio processor', '載入音訊處理'), ('Generating video', '生成畫面與合成音訊')]
                phase = '啟動模型'
                for marker, label in phases:
                    if marker in text: phase = label
                elapsed = int(time.monotonic() - started)
                yield f'{phase} · 已經過 {elapsed} 秒\n外接硬碟首次載入曾約需 13 分鐘；本版每次工作都會重新載入。', None, None, None
                time.sleep(1)
        if CANCEL.is_set():
            yield '已取消，未完成的檔案保留在本機工作資料夾。', None, None, str(log_path)
            return
        if proc.returncode:
            tail = log_path.read_text(errors='replace').splitlines()[-6:]
            raise RuntimeError('生成失敗：' + '\n'.join(tail))
        output = result_video(job, duration)
        elapsed = round(time.monotonic() - started, 1)
        (job / 'result.json').write_text(json.dumps({'elapsed_seconds': elapsed, 'audio_seconds': duration, 'frames': frames, 'steps': PRESETS[quality]}, indent=2))
        yield f'完成 · 總耗時 {elapsed} 秒 · 音訊 {duration:.2f} 秒\n已檢查影音軌與長度。請播放確認嘴型與自然度；此檢查不代表嘴型準確率。', output, output, str(log_path)
    except Exception as exc:
        yield str(exc), None, None, str(job / 'run.log') if job and (job / 'run.log').exists() else None
    finally:
        if proc is not None and proc.poll() is None:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()
        with STATE_LOCK: PROCESS = None
        release_job()
