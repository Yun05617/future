# Browser Voice Recorder Design

The showcase page replaces the visually compressed Gradio recorder with a single custom browser control. It uses MediaRecorder to capture microphone audio as WebM. First press begins capture and changes the icon to stop; second press finalizes the blob and writes it into the existing hidden Gradio audio file input, dispatching its normal change event. The existing Python analysis and send flow remain unchanged.

The recorder must show a clear idle, recording, permission-denied, and recorded state. It uses only browser APIs and local Gradio upload handling. The input dock keeps the upload plus button and send button.
