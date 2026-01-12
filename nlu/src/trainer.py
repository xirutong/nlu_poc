from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from transformers import Trainer

@dataclass
class JointCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 提取意图标签，直接转换为长整型张量。形状: (batch_size)
        intent_labels = torch.tensor([f["intent_labels"] for f in features], dtype=torch.long)
        
        # 2. 提取槽位标签列表（每个样本的槽位序列长度可能不同）
        slot_labels = [f["slot_labels"] for f in features]

        # 3. 准备文本输入数据，剔除掉标签字段，只保留 input_ids, attention_mask 等
        to_pad = []
        for f in features:
            f2 = {k: v for k, v in f.items() if k not in ["intent_labels", "slot_labels"]}
            to_pad.append(f2)

        # 4. 核心步骤：使用分词器对文本序列进行自动填充。
        # 使当前 Batch 内所有序列长度一致，对齐到本 Batch 的最长长度。
        batch = self.tokenizer.pad(to_pad, padding=True, return_tensors="pt")
        
        # 获取填充后的最大长度，用于后续同步填充槽位标签
        max_len = batch["input_ids"].shape[1]

        # 5. 手动填充槽位标签（Slot Labels）
        # 因为槽位标签是 Token 级别的，必须与 input_ids 的长度严格一致
        padded_slot = []
        for sl in slot_labels:
            if len(sl) < max_len:
                # 使用 -100 进行填充，因为 PyTorch 的 CrossEntropyLoss 默认忽略 -100 的计算
                sl = sl + [-100] * (max_len - len(sl))
            else:
                # 如果超出截断（通常由分词器的截断策略保证不会超出）
                sl = sl[:max_len]
            padded_slot.append(sl)

        # 6. 将处理好的标签存入 batch 字典中返回
        batch["intent_labels"] = intent_labels
        batch["slot_labels"] = torch.tensor(padded_slot, dtype=torch.long)
        return batch
    
class JointTrainer(Trainer):
    # 重写计算 Loss 的方法
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 前向传播：将 batch 数据（包括 labels）输入模型
        # 注意：模型内部需要能接收 intent_labels 和 slot_labels 并计算总 Loss
        outputs = model(**inputs)
        
        # 2. 获取模型计算出的联合 Loss
        # 通常是：Loss = weight1 * intent_loss + weight2 * slot_loss
        loss = outputs["loss"]
        
        # 根据 Trainer 规范返回 Loss
        return (loss, outputs) if return_outputs else loss
    
def preprocess_logits_for_metrics(logits, labels):
    """
    此函数用于在 evaluation 循环中减少内存占用。
    它将模型原始输出（通常包含多个张量）转换为评估函数需要的形式。
    """
    # 如果模型返回的是字典（包含意图和槽位两组预测分值）
    if isinstance(logits, dict):
        # 同时返回意图预测值和槽位预测值
        return (logits["intent_logits"], logits["slot_logits"])
    
    # 否则直接返回原始 logits
    return logits