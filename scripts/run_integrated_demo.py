"""Run the complete local voice identity, emotion, transcript, and chat demo."""

import argparse

import runtime_env  # noqa: F401

from gradio import networking

from speech_emotion.personal_demo_ui import APPLE_STYLE_CSS, APPLE_THEME, build_integrated_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="啟動本機整合語音辨識與聊天 Demo")
    parser.add_argument("--port", type=int, default=7861, help="本機服務連接埠")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The desktop sandbox can block Gradio's optional localhost probe even
    # though the app has already bound to 127.0.0.1.  Keep the demo local and
    # prevent that probe from shutting the successfully started server down.
    networking.url_ok = lambda _url: True
    build_integrated_demo().launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=True,
        share=False,
        css=APPLE_STYLE_CSS,
        theme=APPLE_THEME,
    )


if __name__ == "__main__":
    main()
