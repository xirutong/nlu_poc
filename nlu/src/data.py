from typing import Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

def load_snips_joint_dataset(name: str = "bkonkle/snips-joint-intent", val_ratio: float = 0.1, seed: int = 42):
    """
    直接加载 Hugging Face 数据集。
    由于原始数据集只有 train 和 test，我们从 train 中切分一部分作为 validation。
    """
    ds = load_dataset(name)
    
    # 原始数据集结构通常是 ds['train'] 和 ds['test']
    # 我们利用 train_test_split 将 ds['train'] 进一步分为 训练集 和 验证集
    train_val_split = ds["train"].train_test_split(test_size=val_ratio, seed=seed)
    
    # 返回：训练集, 验证集, 测试集
    return train_val_split["train"], train_val_split["test"], ds["test"]

def build_label_maps(train_ds) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """从训练集中提取唯一的意图(Intent)和槽位(Slot)标签，建立 ID 映射表"""
    # 处理意图标签
    intents = sorted(set(train_ds["intent"]))
    intent2id = {x: i for i, x in enumerate(intents)}
    id2intent = {i: x for x, i in intent2id.items()}

    # 处理槽位标签（SNIPS 的槽位通常是以空格分隔的字符串）
    slot_set = set()
    for s in train_ds["slots"]:
        slot_set.update(s.split())
    slots = sorted(slot_set)
    slot2id = {x: i for i, x in enumerate(slots)}
    id2slot = {i: x for x, i in slot2id.items()}
    
    return intent2id, id2intent, slot2id, id2slot

def get_tokenizer(model_name: str):
    """初始化预训练分词器"""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_and_align_factory(tokenizer, intent2id, slot2id, max_length=None):
    """闭包函数：用于在 Dataset.map 中处理文本并对齐槽位标签"""
    def _fn(example):
        words = example["input"].split()
        tags = example["slots"].split()

        # 健壮性检查：确保词和标签数量一致
        if len(words) != len(tags):
            m = min(len(words), len(tags))
            words, tags = words[:m], tags[:m]

        # 分词：is_split_into_words=True 告诉分词器输入已经是 list 类型
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        # 获取每个 token 对应的原始单词索引 (word_ids)
        # 例如: "play music" -> [None, 0, 1, None] (None 代表 [CLS] 或 [SEP])
        word_ids = enc.word_ids()
        aligned = []
        prev = None
        
        for wi in word_ids:
            if wi is None:
                # 特殊 Token (如 [CLS], [SEP]) 标签设为 -100，计算 Loss 时会忽略
                aligned.append(-100)
            elif wi != prev:
                # 单词的第一个 Token，分配对应的槽位 ID
                aligned.append(slot2id[tags[wi]])
            else:
                # 单词被切分后的后续 Subwords（子词），也设为 -100
                aligned.append(-100)
            prev = wi

        # 添加处理后的标签到字典
        enc["intent_labels"] = intent2id[example["intent"]]
        enc["slot_labels"] = aligned
        return enc
    return _fn

if __name__ == "__main__":
    # 测试加载数据集
    train_ds, val_ds, test_ds = load_snips_joint_dataset()

    print("===== 数据集划分情况 =====")
    print(f"训练集 (Train) 大小: {len(train_ds)}")
    print(f"验证集 (Val)   大小: {len(val_ds)}")
    print(f"测试集 (Test)  大小: {len(test_ds)}")

    # 查看数据的列名（Features）
    print(f"\n数据集包含的字段: {train_ds.column_names}")
    
    # 抽取查看样本
    import random

    def preview_raw_data(dataset, num_samples=2):
        print(f"\n===== 随机预览 {num_samples} 条原始数据 =====")
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            item = dataset[idx]
            print(f"ID: {idx}")
            print(f"  文本 (Input): {item['input']}")
            print(f"  意图 (Intent): {item['intent']}")
            print(f"  槽位 (Slots): {item['slots']}")
            print("-" * 20)

    preview_raw_data(train_ds)

    # 查看标签分布
    from collections import Counter
    def check_intent_distribution(dataset):
        intents = dataset["intent"]
        counts = Counter(intents)
        print("\n===== 意图标签分布 =====")
        for intent, count in counts.most_common():
            print(f"{intent:<25}: {count} ({count/len(dataset):.2%})")

    check_intent_distribution(train_ds)
