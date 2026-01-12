import pyaudio
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from typing import Callable, Optional


def load_model(
    model_path: str = "F:/Projects/nlu_poc/model/SenseVoiceSmall",
    device: str = "cuda",
    disable_update: bool = True,
):
    """加载 SenseVoiceSmall 模型并返回模型实例。

    参数:
        model_path: 模型目录路径
        device: 运行设备，例如 'cuda' 或 'cpu'
        disable_update: 是否禁用模型更新
    返回:
        funasr.AutoModel 实例
    """
    return AutoModel(model=model_path, device=device, disable_update=disable_update)


class SenseVoiceSmallRecognizer:
    """基于麦克风的连续语音识别器。

    使用示例:
        recognizer = SenseVoiceSmallRecognizer()
        recognizer.start_listening(lambda text: print('识别:', text))
    """

    def __init__(
        self,
        model: Optional[AutoModel] = None,
        model_path: str = "F:/Projects/nlu_poc/asr/model/SenseVoiceSmall",
        device: str = "cuda",
        disable_update: bool = True,
        chunk: int = 1024,
        rate: int = 16000,
        channels: int = 1,
        format=pyaudio.paInt16,
        silence_threshold: int = 500,
        silence_duration: float = 1,
    ):
        self.model = model or load_model(model_path, device, disable_update)

        # 音频参数
        self.CHUNK = chunk
        self.RATE = rate
        self.CHANNELS = channels
        self.FORMAT = format
        self.SILENCE_THRESHOLD = silence_threshold
        self.SILENCE_DURATION = silence_duration

        # PyAudio 变量，延后到 start_listening 时打开
        self._p = None
        self._stream = None
        self._running = False

    def _open_stream(self):
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._p is not None:
            try:
                self._p.terminate()
            except Exception:
                pass
            self._p = None

    def _transcribe_frames(self, frames: list) -> str:
        audio_buffer = b"".join(frames)
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        res = self.model.generate(input=audio_np, cache={}, language="auto", use_itn=True)
        text = rich_transcription_postprocess(res[0]["text"])
        return text

    def listen_once(self) -> Optional[str]:
        """读取麦克风直到检测到语句结束，然后返回识别文本（若无则返回 None）。"""
        if self._stream is None:
            raise RuntimeError("音频流未打开，请先调用 start_listening 或 _open_stream")

        frames = []
        silent_chunks = 0
        speaking = False

        while True:
            data = self._stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            amplitude = np.max(np.abs(audio_data))

            if amplitude > self.SILENCE_THRESHOLD:
                speaking = True
                silent_chunks = 0
            else:
                if speaking:
                    silent_chunks += 1

            if speaking and (silent_chunks > int(self.SILENCE_DURATION * self.RATE / self.CHUNK)):
                break

        if frames:
            return self._transcribe_frames(frames)
        return None

    def start_listening(self, callback: Callable[[str], None]):
        """开始监听麦克风并在每次识别到句子时调用 callback(text)。

        参数:
            callback: 接收识别结果的函数，签名为 callback(text: str)
        """
        self._open_stream()
        self._running = True
        print(">>> 正在监听麦克风... (按 Ctrl+C 退出)")

        try:
            while self._running:
                text = self.listen_once()
                if text:
                    callback(text)
                    print("-" * 20)
                    if "再见" in text:
                        print("\n检测到退出指令，停止监听。")
                        break
        except KeyboardInterrupt:
            print("\n用户退出，已停止监听。")
        finally:
            self._running = False
            self._close_stream()

    def stop(self):
        self._running = False
        self._close_stream()

    def close(self):
        self.stop()


__all__ = ["load_model", "SenseVoiceSmallRecognizer"]


def main():
    """命令行运行示例，保持原脚本的行为（打印识别结果）。"""
    recognizer = SenseVoiceSmallRecognizer()

    def _print_cb(text: str):
        print(f"识别结果: {text}")

    recognizer.start_listening(_print_cb)


if __name__ == "__main__":
    main()