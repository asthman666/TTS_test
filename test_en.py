import torch
import torchaudio as ta

# Monkey-patch torch.load to use map_location for CPU before importing Chatterbox
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model (with cache cleared above)
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

# Generate with Paralinguistic Tags
text = "Amazon Bedrock also offers a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top foundation models for your use cases, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation, and build agents that execute tasks using your enterprise systems and data sources."

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, audio_prompt_path="NAN_en.m4a")

ta.save("en-nan.wav", wav, model.sr)
