import glob
import json
import os
import torch

from nlu.src.data import get_tokenizer
from nlu.src.model import JointIntentSlotModel

def main(text2predict: str):
    # 1. 配置路径：从环境变量获取 Checkpoint 路径，默认指向最后一次保存的目录
    base_dir = "nlu/outputs/snips_joint"
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoints:
        print(f"错误：在 {base_dir} 未找到任何 checkpoint 文件夹！")
        return
    
    ckpt = max(checkpoints, key=lambda p: int(os.path.basename(p).split("-")[-1]))
    print(f"正在加载最强/最新的权重: {ckpt}")
    
    model_name = os.environ.get("MODEL_NAME", "distilbert-base-uncased")

    # 2. 初始化分词器
    tokenizer = get_tokenizer(model_name)

    # 3. 初始化模型结构
    # 注意：这里的 num_intents 和 num_slots 必须与训练时完全一致，否则权重矩阵形状对不上
    model = JointIntentSlotModel(base_model_name=model_name, num_intents=7, num_slots=72)
    
    # 4. 加载权重
    # 使用 map_location="cpu" 确保即使在没有 GPU 的机器上也能加载模型
    weights_path = os.path.join(ckpt, "model.safetensors")
    if os.path.exists(weights_path):
        # 注意：safetensors 通常使用 safetensors 库加载，或者模型类自带 .from_pretrained
        # 如果是手动 load_state_dict，可以使用以下方式转换
        from safetensors.torch import load_file
        state = load_file(weights_path)
        model.load_state_dict(state, strict=False)
        print("成功加载 model.safetensors")
    else:
        # 尝试加载旧版 .bin
        weights_path = os.path.join(ckpt, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("成功加载 pytorch_model.bin")
    
    # 设置为评估模式：关闭 Dropout，固定 Batch Normalization
    model.eval()

    # 5. 测试文本预处理
    words = text2predict.split() # 将句子切分为单词列表
    
    # is_split_into_words=True 告诉分词器：输入已经是切分好的词，不需要再次按空格切分
    # return_tensors="pt" 返回 PyTorch 张量格式
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt")

    # 6. 执行推理
    with torch.no_grad(): # 推理阶段不需要计算梯度，节省显存并提速
        out = model(**enc)

    # 7. 预览
    # intent_logits 形状: [batch_size, num_intents] -> (1, 10)
    #print("intent logits shape:", out["intent_logits"].shape)
    
    # slot_logits 形状: [batch_size, sequence_length, num_slots] -> (1, seq_len, 50)
    # 注意：这里的 sequence_length 是分词后的长度（包含 [CLS], [SEP] 和子词）
    #print("slot logits shape:", out["slot_logits"].shape)

    # 8. 获取预测结果
    import json
    mapping_path = os.path.join("nlu/outputs/snips_joint", "label_mapping.json") # 加载 label 映射
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    id2intent = mapping["id2intent"]
    id2slot = mapping["id2slot"]
    
    intent_idx = torch.argmax(out["intent_logits"], dim=-1).item()
    slot_indices = torch.argmax(out["slot_logits"], dim=-1).squeeze(0).tolist()
    
    # 8.1 获取意图文字
    intent_name = id2intent[str(intent_idx)] # 注意：json加载后的key通常是字符串
    print(f"\n预测意图: {intent_name}")

    # 8.2 获取槽位文字并与单词对应
    # 获取 Token 对应的单词索引，例如 [None, 0, 1, 1, 2, None]
    word_ids = enc.word_ids() 

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    words = text2predict.split() # 原始输入的单词列表

    print(f"{'Token':<15} | {'Word_ID':<8} | {'Predict_Slot':<15}")
    print("-" * 45)

    for i, slot_id in enumerate(slot_indices):
        word_idx = word_ids[i]
        token_str = tokens[i]
        slot_name = id2slot[str(slot_id)]
        
        # 打印所有 Token 的预测情况（包括子词和特殊字符）
        print(f"{token_str:<15} | {str(word_idx):<8} | {slot_name:<15}")

    # 8.3 更有用的展示：只显示原始单词对应的槽位 (过滤特殊字符和子词)
    print("\n===== 单词级结果 =====")
    last_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != last_word_idx:
            # 只取每个单词的第一个 Token 的预测值
            slot_name = id2slot[str(slot_indices[i])]
            print(f"单词: {words[word_idx]:<10} -> 标签: {slot_name}")
            last_word_idx = word_idx

if __name__ == "__main__":
    main(text2predict="what is the weather like in munich tomorrow")