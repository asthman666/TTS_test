import torch
import torchaudio as ta

# Monkey-patch torch.load to use map_location for CPU before importing Chatterbox
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cpu')
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

chinese_text = "没有利润，企业就无法存在。获取利润并不是注定错误的，但利润必须依靠良好的服务而获得，诚实经营的商业企业不可能得不到利润回报。利润不能是基础，它必须是服务的结果。"
AUDIO_PROMPT_PATH = "SHUN.m4a"
wav_chinese2 = multilingual_model.generate(chinese_text, language_id="zh", audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("zh-shun.wav", wav_chinese2, multilingual_model.sr)
