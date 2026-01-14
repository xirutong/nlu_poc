"""
本文档展示了一个用语音进行意图识别和槽位填充的NLU的pipeline代码演示。
目前包含的模块有：
1. ASR模块：将语音转化为文本。
2. NLU模块：对文本进行意图识别和槽位填充。
"""

from asr.sense_voice_small import SenseVoiceSmallRecognizer
from nlu.inference.predictor import NLUModel

asr_model = SenseVoiceSmallRecognizer()
nlu_model = NLUModel()

def asr_to_nlu_cb(text: str):
    text = (text or "").strip()
    if not text:
        return
    nlu_model.load()
    nlu_model.predict(text)

asr_model.start_listening(asr_to_nlu_cb)
