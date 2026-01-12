import torch.nn as nn
from transformers import AutoModel

class JointIntentSlotModel(nn.Module):
    def __init__(self, base_model_name: str, num_intents: int, num_slots: int, dropout: float = 0.1):
        """
        联合意图识别和槽位填充模型
        :param base_model_name: 预训练模型名称 (如 'bert-base-uncased')
        :param num_intents: 意图分类的数量
        :param num_slots: 槽位分类的数量 (包括特殊标签如 O, B-xxx, I-xxx)
        """
        super().__init__()
        # 加载预训练的 Encoder (如 BERT, RoBERTa 等)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size # 提取隐藏层维度，BERT 通常是 768

        self.dropout = nn.Dropout(dropout)
        
        # 意图分类器：输入 [CLS] 的向量，输出意图类别
        self.intent_classifier = nn.Linear(hidden, num_intents)
        
        # 槽位分类器：输入每个 token 的向量，输出对应的槽位标签
        self.slot_classifier = nn.Linear(hidden, num_slots)

        # 损失函数定义
        self.intent_loss_fct = nn.CrossEntropyLoss()
        # 槽位损失需要忽略标签为 -100 的位置（这些通常是 [CLS], [SEP] 或 Subwords）
        self.slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        intent_labels=None,
        slot_labels=None,
        **kwargs
    ):
        """
        前向传播
        :param input_ids: [Batch_Size, Seq_Length]
        :param attention_mask: [Batch_Size, Seq_Length]
        :param intent_labels: [Batch_Size]
        :param slot_labels: [Batch_Size, Seq_Length]
        """
        # 1. 经过基础模型编码
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # 获取所有 token 的最后隐层状态: [B, T, H] (Batch, Time/Seq, Hidden)
        seq = self.dropout(out.last_hidden_state)  

        # 2. 意图分类：取序列的第一个 Token (通常是 [CLS]) 作为句子的整体表示
        # seq[:, 0, :] 维度为 [Batch_Size, Hidden]
        pooled = seq[:, 0, :]                             
        intent_logits = self.intent_classifier(pooled)    # 结果维度: [B, num_intents]

        # 3. 槽位分类：对序列中的每一个 token 进行分类
        # seq 维度为 [B, T, H]，slot_logits 维度为 [B, T, num_slots]
        slot_logits = self.slot_classifier(seq)            

        loss = None
        # 4. 如果提供了标签，则计算 Loss
        if intent_labels is not None and slot_labels is not None:
            # 意图分类损失
            intent_loss = self.intent_loss_fct(intent_logits, intent_labels)
            
            # 槽位填充损失：需要把 [B, T, S] 展平为 [B*T, S] 以匹配交叉熵输入
            slot_loss = self.slot_loss_fct(
                slot_logits.view(-1, slot_logits.size(-1)),
                slot_labels.view(-1),
            )
            
            # 联合损失（你也可以给两个 loss 设置权重，例如 loss = a * i_loss + b * s_loss）
            loss = intent_loss + slot_loss

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }