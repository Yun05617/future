#!/bin/bash
# 放在 future-main 根目錄，雙擊即可。首次啟動會安裝 requirements.txt。
set -Eeuo pipefail

finish() {
    result=$?
    trap - EXIT
    if [ "$result" -ne 0 ]; then
        printf '\n啟動未完成（代碼 %s），請查看上方錯誤。\n' "$result"
    fi
    if [ -t 0 ]; then
        printf '\n按 Enter 關閉視窗。'
        read -r _ || true
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
cd "$ROOT"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ "$(uname -s)" != Darwin ]; then
    echo '此啟動器僅供 macOS 使用。' >&2
    exit 1
fi
for file in requirements.txt scripts/prepare_handoff.py scripts/run_competition_demo.py; do
    if [ ! -f "$file" ]; then
        echo '請先將此檔案放入 future-main 根目錄（與 start_windows.bat 同一層）。' >&2
        exit 1
    fi
done

# 與 Windows 的 .venv 分開，避免誤用或覆寫其他平台的環境。
VENV="$ROOT/.venv-macos"
PYTHON="$VENV/bin/python"
if [ ! -x "$PYTHON" ]; then
    BASE=''
    for candidate in python3.12 /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/opt/python@3.12/bin/python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
            BASE="$candidate"
            break
        fi
    done
    if [ -z "$BASE" ]; then
        echo '需要 Python 3.12。若已安裝 Homebrew，請執行：brew install python@3.12' >&2
        exit 1
    fi
    "$BASE" -m venv "$VENV"
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "此專案需要 Python 3.12"'
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$ROOT/.matplotlib"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export GRADIO_ANALYTICS_ENABLED=False
export VOICE_PROJECT_WORKSPACE="${VOICE_PROJECT_WORKSPACE:-$ROOT/workspace}"

# 先檢查連接埠，避免重複啟動；不會終止既有程序。
"$PYTHON" - <<'PY'
import socket
for port in (7860, 7862, 7876):
    with socket.socket() as listener:
        try:
            listener.bind(('127.0.0.1', port))
        except OSError as exc:
            raise SystemExit(f'連接埠 {port} 無法使用：{exc}。請先關閉先前啟動的專題。')
PY

STAMP="$VENV/.requirements-installed"
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
    echo '首次使用或套件清單已更新，正在安裝套件；可能需要較長時間。'
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r requirements.txt
    cp requirements.txt "$STAMP"
fi

# Windows 的 ffmpeg.exe 不能在 Mac 執行；使用已安裝的原生版本，
# 或 requirements.txt 已包含的 imageio-ffmpeg 所附 macOS 執行檔。
if ! command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG="$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
    if [ ! -x "$FFMPEG" ]; then
        echo '找不到可執行的 macOS FFmpeg。請安裝：brew install ffmpeg' >&2
        exit 1
    fi
    ln -sf "$FFMPEG" "$VENV/bin/ffmpeg"
fi
ffmpeg -version >/dev/null

# 嘗試開啟已安裝的 Ollama；不自動下載大型模型。
if ! curl --silent --fail --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null; then
    open -a Ollama >/dev/null 2>&1 || true
    echo '已嘗試開啟 Ollama；聊天前請確認 Ollama 正在執行，且已安裝 gemma4:12b。'
fi

"$PYTHON" "$ROOT/scripts/prepare_handoff.py"
echo '即將開啟完整整合頁面：http://127.0.0.1:7862/'
echo '訓練後台與聲音擴增服務會在背景一併啟動。'
echo '請保持此視窗開啟；按 Control+C 停止全部介面。'
"$PYTHON" "$ROOT/scripts/run_competition_demo.py"
