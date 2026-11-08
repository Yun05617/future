# Data and Model Provenance

Record every dataset, model, and external package that will appear in a competition build before using it.

This is a historical provenance record, not a current competition-compliance certification. Historical revisions below may differ from a user's downloaded cache. The identity feature cache records its actual local Hugging Face revision; downloaded weights and private training data are never included in this release. Recheck the competition's precise origin and data rules before use.

| Item | Type | Version | Source URL | Licence | Allowed use checked on | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Pilot recordings made by team members or volunteers | Primary speech data | Pending collection | Internal collection with signed consent | Team consent form | 2026-09-21 | Use only after each speaker agrees to the project purpose and the recording is added to the manifest. Do not upload raw voice recordings to the repository. |
| Meta XLS-R 300M (`facebook/wav2vec2-xls-r-300m`) | Frozen multilingual speech feature extractor | 300M parameters; revision `1a640f32ac3e39899438a2931f9924c02f080a54` | https://huggingface.co/facebook/wav2vec2-xls-r-300m | Apache-2.0 | 2026-09-21 | Candidate for the competition build: official model card identifies Meta as source, 128-language pretraining, and Apache-2.0 licensing. Downloaded and validated locally on Apple Silicon MPS. |
| scikit-learn | Classifier library | 1.9.1 | https://scikit-learn.org/ | BSD-3-Clause | 2026-09-21 | Allowed as the local classifier library. |
| SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | Speaker embedding model | historical revision `eac27266f68caa806381260bd44ace38b136c76a`; SpeechBrain 1.1.1 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | Apache-2.0 | 2026-09-22 | Local speaker embeddings. Check the model, dependencies, and pretraining-data origin against the competition rules; do not infer blanket eligibility from the publisher alone. No cloud inference. |
| OpenAI Whisper | Chinese speech-to-text model | openai-whisper 20250625 | https://github.com/openai/whisper | MIT | 2026-09-22 | Candidate for local Chinese transcription: OpenAI's project releases code and weights under MIT. The app fixes transcription language to Chinese and does not send audio to an external API. |

No external speech dataset is approved for this project yet. Add its source URL, licence terms, competition eligibility check, and decision here before downloading or training on it.
