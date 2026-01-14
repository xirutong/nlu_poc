import os
import numpy as np
import torch
import evaluate
from transformers import TrainingArguments, set_seed
import argparse
from settings import load_settings
settings = load_settings()

from nlu.src.data import (
    load_snips_joint_dataset,
    build_label_maps,
    get_tokenizer,
    tokenize_and_align_factory,
)
from nlu.src.model import JointIntentSlotModel
from nlu.src.trainer import JointTrainer, JointCollator, preprocess_logits_for_metrics
from nlu.utils.get_training_plots import save_training_plots


def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="Disable saving training plots")
    cli_args = parser.parse_args()

    base_model_name = settings.nlu.base_model_name

    # 加载 SNIPS
    train_ds, val_ds, test_ds = load_snips_joint_dataset(val_ratio=0.1, seed=42)
    
    # 构建 标签<->ID 的映射表。例如：'PlayMusic' -> 0, 'B-artist' -> 1
    intent2id, id2intent, slot2id, id2slot = build_label_maps(train_ds)

    tokenizer = get_tokenizer(base_model_name)
    # 这是一个工厂函数，生成的 tok_fn 会处理：
    # 1. 分词 2. 将词级标签对齐到 Subword（例如 "looking" 被切分为 "look", "##ing" 时，标签也要复制）
    tok_fn = tokenize_and_align_factory(tokenizer, intent2id, slot2id)

    # 对整个数据集进行映射处理，并移除原始文本列，只保留模型需要的 input_ids 等张量
    train_tok = train_ds.map(tok_fn, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tok_fn, remove_columns=val_ds.column_names)
    test_tok = test_ds.map(tok_fn, remove_columns=test_ds.column_names)

    # 初始化自定义的联合模型
    model = JointIntentSlotModel(
        base_model_name= base_model_name,
        num_intents=len(intent2id), # 顶层的意图分类维度
        num_slots=len(slot2id),      # 每个 Token 的槽位分类维度
        dropout=0.1,
    )

    # 加载评估工具：accuracy 算意图，seqeval 是序列标注（槽位）的标准度量工具
    acc = evaluate.load("accuracy")
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        # 解包预测值和真实标签，这些是由 preprocess_logits_for_metrics 预处理后的格式
        (intent_logits, slot_logits), (intent_labels, slot_labels) = eval_pred

        # --- 意图计算 ---
        intent_pred = np.argmax(intent_logits, axis=-1)
        intent_metrics = acc.compute(predictions=intent_pred, references=intent_labels)

        # --- 槽位计算 ---
        slot_pred = np.argmax(slot_logits, axis=-1)

        true_predictions, true_labels = [], []
        for p_seq, l_seq in zip(slot_pred, slot_labels):
            p_tags, l_tags = [], []
            for p_id, l_id in zip(p_seq, l_seq):
                # 关键：跳过 -100 (Padding 或特殊 Token)，不计入 F1 指标
                if l_id == -100:
                    continue
                p_tags.append(id2slot[int(p_id)])
                l_tags.append(id2slot[int(l_id)])
            true_predictions.append(p_tags)
            true_labels.append(l_tags)

        # seqeval 会计算 BIO 格式下的精确率、召回率和 F1
        slot_metrics = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "intent_accuracy": intent_metrics["accuracy"],
            "slot_f1": slot_metrics.get("overall_f1", 0.0),
            "slot_precision": slot_metrics.get("overall_precision", 0.0),
            "slot_recall": slot_metrics.get("overall_recall", 0.0),
        }
    
    args = TrainingArguments(
        output_dir=settings.paths.model_dir,
        eval_strategy="epoch",    # 每个 Epoch 结束后跑一次验证集
        save_strategy="epoch",    # 每个 Epoch 结束后保存一次模型
        logging_steps=10,
        logging_first_step=True,
        learning_rate=1e-5,       # 预训练模型微调通常使用较小的学习率
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True, # 训练结束自动加载验证集上表现最好的模型权重
        metric_for_best_model="slot_f1", # 以槽位 F1 值作为评价“最好”的标准
        greater_is_better=True,
        fp16=torch.cuda.is_available(), # 如果有 GPU，开启混合精度训练以节省显存并提速
        report_to="none",
    )

    trainer = JointTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=JointCollator(tokenizer), # 负责把不同长度的句子和标签对齐成 batch
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # 预处理输出格式
    )
    # 明确告知 Trainer，Dataset 中哪几列是我们要传入模型 forward 函数的标签
    trainer.label_names = ["intent_labels", "slot_labels"]

    # 1. 执行训练
    print("1️⃣开始训练=========================================================>")
    trainer.train() 

    # 2. 保存标签映射表 (关键：为了 predict.py 的稳健性)
    import json
    print("2️⃣训练结束，准备保存 mapping=========================================================>")
    full_output_dir = os.path.abspath(args.output_dir) # 获取绝对路径
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir, exist_ok=True)

    mapping = {"id2intent": id2intent, "id2slot": id2slot}
    target_path = os.path.join(full_output_dir, "label_mapping.json")

    try:
        with open(target_path, "w", encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        print(f"成功保存 mapping 到: {target_path}")
    except Exception as e:
        print(f"保存 mapping 失败，错误原因: {e}")

    # 3. 生成可视化图表
    print("3️⃣准备生成训练图=========================================================>")
    if not cli_args.no_plot:
        save_training_plots(trainer, args.output_dir)
    
    # 4. 最终测试
    print("4️⃣输出最终测试结果=========================================================>")
    print("Test Results:", trainer.evaluate(test_tok))


if __name__ == "__main__":
    main()
