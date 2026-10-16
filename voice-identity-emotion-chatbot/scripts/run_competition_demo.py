"""Start the local training back office and competition showcase together."""

import webbrowser

import runtime_env  # noqa: F401

from gradio import networking

from speech_emotion.personal_demo_ui import APPLE_STYLE_CSS, APPLE_THEME, build_demo
from speech_emotion.showcase_ui import SHOWCASE_CSS, build_showcase_demo


def main() -> None:
    """Keep both local pages available so their navigation links work."""
    # The desktop app can block Gradio's optional localhost probe even after
    # binding succeeds.  Both pages remain local-only.
    networking.url_ok = lambda _url: True

    build_demo().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        share=False,
        prevent_thread_lock=True,
        css=APPLE_STYLE_CSS,
        theme=APPLE_THEME,
    )
    build_showcase_demo().launch(
        server_name="127.0.0.1",
        server_port=7862,
        inbrowser=False,
        share=False,
        css=SHOWCASE_CSS,
    )
    # Open both local pages after both servers have been started.  This keeps
    # the one-command workflow while still giving the user two separate tabs.
    webbrowser.open_new_tab("http://127.0.0.1:7860/")
    webbrowser.open_new_tab("http://127.0.0.1:7862/")
    print("模型訓練後台：http://127.0.0.1:7860/")
    print("競賽展示頁面：http://127.0.0.1:7862/")


if __name__ == "__main__":
    main()
