"""
本文档展示了一个用语音进行意图识别和槽位填充的NLU的pipeline代码演示。
目前包含的模块有：
1. ASR模块：将语音转化为文本。
2. NLU模块：对文本进行意图识别和槽位填充。
"""

from asr.sense_voice_small import SenseVoiceSmallRecognizer
from nlu.scripts.predict import main as nlu_predict


# 1. 执行ASR模块
asr_model = SenseVoiceSmallRecognizer()
def _print_cb(text: str):
        print(f"==============================ASR识别结果==============================")
        print(text)   
asr_model.start_listening(_print_cb)

# 2. 执行NLU模块
