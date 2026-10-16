"""Run the competition showcase page separately from the training back office."""

import argparse

import runtime_env  # noqa: F401

from speech_emotion.showcase_ui import SHOWCASE_CSS, build_showcase_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="啟動具身分辨識與情緒感知的聊天機器人競賽展示主頁")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    build_showcase_demo().launch(
        server_name="127.0.0.1", server_port=args.port, inbrowser=True, share=False,
        css=SHOWCASE_CSS,
    )


if __name__ == "__main__":
    main()
