"""Start training, showcase, and audio augmentation together."""

import webbrowser

import runtime_env  # noqa: F401

import gradio as gr
from gradio import networking

from speech_emotion.personal_demo_ui import APPLE_STYLE_CSS, APPLE_THEME, build_demo
from speech_emotion.showcase_ui import SHOWCASE_CSS, build_showcase_demo
from audio_augmentation_tool import CSS as AUGMENTATION_CSS, build_app as build_augmentation_app


def main() -> None:
    """Start every local interface before blocking or opening browser tabs."""
    # The desktop app can block Gradio's optional localhost probe even after
    # binding succeeds. All three interfaces remain local-only.
    networking.url_ok = lambda _url: True

    training = build_demo()
    showcase = build_showcase_demo()
    augmentation = build_augmentation_app().queue()
    started = []
    try:
        training.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=False,
            share=False,
            prevent_thread_lock=True,
            css=APPLE_STYLE_CSS,
            theme=APPLE_THEME,
        )
        started.append(training)
        showcase.launch(
            server_name="127.0.0.1",
            server_port=7862,
            inbrowser=False,
            share=False,
            prevent_thread_lock=True,
            css=SHOWCASE_CSS,
        )
        started.append(showcase)
        augmentation.launch(server_name="127.0.0.1", server_port=7876,
                            inbrowser=False, share=False, prevent_thread_lock=True,
                            css=AUGMENTATION_CSS, theme=gr.themes.Default(primary_hue="teal"))
        started.append(augmentation)
        for url in ("http://127.0.0.1:7860/", "http://127.0.0.1:7862/", "http://127.0.0.1:7876/"):
            webbrowser.open_new_tab(url)
            print("Ready: " + url, flush=True)
        showcase.block_thread()
    finally:
        for app in reversed(started):
            app.close()


if __name__ == "__main__":
    main()
