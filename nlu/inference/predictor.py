import glob
import json
import os
from typing import Dict, List, Optional, Tuple
import torch
from settings import load_settings
settings = load_settings()  

from nlu.src.data import get_tokenizer
from nlu.src.model import JointIntentSlotModel
from nlu.utils.get_structured_slots import bio_to_structured_slots

class NLUModel:
    """A reusable NLU model wrapper for intent-slot prediction.

    Example usage:
        model = NLUModel()
        model.load(device="cpu")
        res = model.predict("what is the weather like in munich tomorrow")
        print(res["intent"], res["word_slots"]) 
    """

    def __init__(
        self,
        base_dir: str = settings.paths.model_dir,
        model_name: str = settings.nlu.base_model_name,
        num_intents: int = 7,
        num_slots: int = 72,
        mapping_path: Optional[str] = None,
    ) -> None:
        self.base_dir = base_dir
        self.model_name = model_name
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.mapping_path = mapping_path or os.path.join(base_dir, "label_mapping.json")

        self.tokenizer = None
        self.model = None
        self.id2intent = {}
        self.id2slot = {}
        self.ckpt = None

    def _find_latest_checkpoint(self) -> str:
        checkpoints = glob.glob(os.path.join(self.base_dir, "checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.base_dir}")
        ckpt = max(checkpoints, key=lambda p: int(os.path.basename(p).split("-")[-1]))
        return ckpt

    def load(self, ckpt: Optional[str] = None, device: str = "cpu") -> None:
        """Load tokenizer, model weights and label mapping.

        Args:
            ckpt: Optional explicit checkpoint path. If None, find the latest in base_dir.
            device: Device to move the model to ("cpu" or "cuda").
        """
        self.ckpt = ckpt or self._find_latest_checkpoint()
        print(f"Loading checkpoint: {self.ckpt}")

        # tokenizer
        self.tokenizer = get_tokenizer(self.model_name)

        # model structure
        self.model = JointIntentSlotModel(base_model_name=self.model_name, num_intents=self.num_intents, num_slots=self.num_slots)

        # try safetensors first
        weights_path = os.path.join(self.ckpt, "model.safetensors")
        if os.path.exists(weights_path):
            try:
                from safetensors.torch import load_file

                state = load_file(weights_path)
                self.model.load_state_dict(state, strict=False)
                print("Loaded model.safetensors")
            except Exception as e:
                print("Failed to load safetensors:", e)
        else:
            weights_path = os.path.join(self.ckpt, "pytorch_model.bin")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
                print("Loaded pytorch_model.bin")

        # load mapping
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            self.id2intent = mapping.get("id2intent", {})
            self.id2slot = mapping.get("id2slot", {})
        else:
            print(f"Warning: mapping file not found at {self.mapping_path}. id2intent/id2slot empty.")

        # device
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, split_words: bool = True, preview: bool = True) -> Dict:
        """Predict intent and slots for `text`.

        Args:
            text: input sentence. If `split_words=True`, the sentence is split on whitespace and
                  tokens are treated as already split words (keeps alignment with original script).
            split_words: whether to call tokenizer(..., is_split_into_words=True) with pre-split words.

        Returns:
            A dict with keys: `intent` (dict), `tokens` (list), `token_slots` (list), `word_slots` (list of tuples (word, slot_label)), `word_ids` (list)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if split_words:
            words = text.split()
            enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
        else:
            enc = self.tokenizer(text, return_tensors="pt")

        # move tensors to device
        enc_device = {k: v.to(self.device) for k, v in enc.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            out = self.model(**enc_device)

        # get logits back on CPU
        intent_logits = out["intent_logits"].cpu()
        slot_logits = out["slot_logits"].cpu()

        intent_idx = int(torch.argmax(intent_logits, dim=-1).item())
        slot_indices = torch.argmax(slot_logits, dim=-1).squeeze(0).tolist()

        # tokens and word ids
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        try:
            word_ids = enc.word_ids()
        except Exception:
            # fallback: no word_ids available
            word_ids = [None] * len(tokens)

        token_slots = [self.id2slot.get(str(sid), str(sid)) for sid in slot_indices]

        # build word-level results (take first token per word)
        word_slots: List[Tuple[str, str]] = []
        words_list = text.split() if split_words else None
        last_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != last_word_idx:
                slot_name = token_slots[i]
                word = words_list[word_idx] if words_list is not None else tokens[i]
                word_slots.append((word, slot_name))
                last_word_idx = word_idx

        result = {
            "intent": {"idx": intent_idx, "name": self.id2intent.get(str(intent_idx), str(intent_idx))},
            "tokens": tokens,
            "token_slots": token_slots,
            "word_slots": word_slots,
            "word_ids": word_ids,
            "checkpoint": self.ckpt,
        }

        # convert word_slots to structured format
        structured_slots = bio_to_structured_slots(word_slots_tuple=word_slots)

        if preview:
            print("===== NLU结果预览 =====")
            print(f"\n🗣️ 识别到的文本: {text}")
            print(f"\n🟢 预测意图: {result['intent']['name']}")
            print(f"{'Token':<15} | {'Word_ID':<8} | {'Predict_Slot':<15}")
            print("-" * 45)
            for i, slot_name in enumerate(token_slots):
                print(f"{tokens[i]:<15} | {str(word_ids[i]):<8} | {slot_name:<15}")

            print("\n🔻 单词级结果")
            print("-"*40)
            for word, slot in word_slots:
                print(f"单词: {word:<10} -> 标签: {slot}")

            print("\n✅ 返回的JSON")
            print(structured_slots)
        
        return structured_slots
 