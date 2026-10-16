"""Run the local personal three-emotion Gradio demo."""

import argparse

import runtime_env  # noqa: F401

from speech_emotion.personal_demo_ui import APPLE_STYLE_CSS, APPLE_THEME, build_demo


def parse_args() -> argparse.Namespace:
    """Parse local server options without starting the application."""
    parser = argparse.ArgumentParser(description="啟動本機個人三情緒語音辨識 Demo")
    parser.add_argument("--port", type=int, default=7860, help="本機服務連接埠")
    return parser.parse_args()


def main() -> None:
    """Build once and open the page in the default browser."""
    args = parse_args()
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=True,
        share=False,
        css=APPLE_STYLE_CSS,
        theme=APPLE_THEME,
    )


if __name__ == "__main__":
    main()
