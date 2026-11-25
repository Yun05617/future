"""Run with python -m mac.photo_webui from the EchoMimic root."""
import gradio as gr
from mac.photo_service import ROOT, PRESETS, run_job, cancel_job

CSS = '''
.gradio-container {max-width:1180px!important;margin:auto!important;background:#f6f2eb!important;color:#292622!important;font-family:system-ui,"PingFang TC",sans-serif!important; padding:28px!important}
body {background:#f6f2eb!important}
#hero {padding:24px 28px!important;border:1px solid #ddd4c8!important;border-radius:20px!important;background:#fffaf3!important;margin-bottom:20px!important}
#hero h1 {font-size:30px!important;line-height:1.4!important;color:#292622!important}
#hero p {color:#766d63!important;line-height:1.8!important}
#generate {background:#ad654b!important;color:white!important;border:0!important;border-radius:12px!important;min-height:48px!important}
#status textarea {line-height:1.8!important;background:#fffaf3!important;color:#51473e!important}
footer {display:none!important}
'''

def build_ui():
    theme = gr.themes.Soft(primary_hue="orange", secondary_hue="orange", neutral_hue="stone")
    # This test room deliberately uses one consistent warm palette in either OS mode.
    theme.set(**{key: getattr(theme, key[:-5]) for key in vars(theme) if key.endswith("_dark") and hasattr(theme, key[:-5])})
    with gr.Blocks(title='照片說話測試室', css=CSS, theme=theme) as demo:
        gr.Markdown('# 照片說話測試室\n上傳一張照片，讓它說出你的聲音。\n\n本機 EchoMimic 實驗版 · 素材留在這台電腦 · 生成需要等待', elem_id='hero')
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### 01　選擇照片\n建議使用單人、正面、光線清楚的照片。')
                photo = gr.Image(label='照片預覽', source='upload', type='filepath', height=300)
                gr.Markdown('### 02　準備聲音\n請使用 **0.5～5 秒**的音訊，建議先測試 1～3 秒。')
                audio = gr.Audio(label='上傳音訊', source='upload', type='filepath')
                with gr.Accordion('或使用麥克風錄製', open=False):
                    recording = gr.Audio(label='錄製後按「使用這段錄音」', source='microphone', type='filepath')
                    use_recording = gr.Button('使用這段錄音')
                    use_recording.click(lambda path: path, recording, audio, queue=False)
                quality = gr.Dropdown(label='測試品質', choices=list(PRESETS), value=list(PRESETS)[1])
                gr.Markdown('2 步只檢查流程，可能模糊或變形；10／20 步較慢，品質仍需實測。')
                with gr.Row():
                    generate = gr.Button('生成說話影片', variant='primary', elem_id='generate')
                    cancel = gr.Button('取消目前工作')
            with gr.Column(scale=1):
                gr.Markdown('### 03　觀看結果')
                video = gr.Video(label='本次生成結果', height=360)
                status = gr.Textbox(label='工作狀態', value='等待照片與音訊。\n請保持 KATHY 外接硬碟連接。每次生成會重新載入模型，可能等待十多分鐘。', lines=5, interactive=False, elem_id='status')
                download = gr.File(label='下載本次影片', interactive=False)
                log = gr.File(label='本次工作日誌（失敗時可提供排查）', interactive=False)
                gr.Markdown('**影音一起播放**：結果為含音軌的 MP4。平台核對長度，嘴型是否自然、對得上發音，仍需觀看評估。')
        start = generate.click(lambda: gr.update(interactive=False), outputs=generate, queue=False)
        job_event = start.then(run_job, [photo, audio, quality], [status, video, download, log])
        job_event.then(lambda: gr.update(interactive=True), outputs=generate, queue=False)
        cancel.click(lambda: (cancel_job(), gr.update(interactive=True)), outputs=[status, generate], cancels=[job_event], queue=False)
        with gr.Accordion('先看已完成的官方範例', open=False):
            sample = ROOT / 'output/20261103/1147--seed_420-256x256/a_echomimic_en_256x256_2_1147_withaudio.mp4'
            if sample.exists():
                gr.Video(value=str(sample), label='先前的 2 步範例，非本次新生成', height=260)
            gr.Markdown('這是一秒的低步數測試，不能代表正式畫質。未上傳音訊時，平台不會自動捏造說話內容。')
    return demo

if __name__ == '__main__':
    build_ui().queue(concurrency_count=1, max_size=2).launch(server_name='127.0.0.1', server_port=7881, share=False, inbrowser=False)
