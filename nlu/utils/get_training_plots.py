import matplotlib.pyplot as plt
import os

def save_training_plots(trainer, base_output_dir):
    out_dir = os.path.join(base_output_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    log = trainer.state.log_history
    print("以下为log:", log)

    # 简单指数移动平均平滑函数（alpha 越大，越接近原始值；建议 0.1~0.5）
    def smooth(values, alpha=0.3):
        smoothed = []
        s = None
        for v in values:
            if v is None:
                smoothed.append(None)
                continue
            if s is None:
                s = v
            else:
                s = alpha * v + (1 - alpha) * s
            smoothed.append(s)
        return smoothed
    
    # Step-level loss（若有）
    steps = [e["step"] for e in log if "step" in e and "loss" in e]
    step_losses = [e["loss"] for e in log if "step" in e and "loss" in e]
    if steps and step_losses:
        sm_step_losses = smooth(step_losses)
        xs = [s for s, v in zip(steps, sm_step_losses) if v is not None]
        ys = [v for v in sm_step_losses if v is not None]
        fig = plt.figure()
        plt.plot(xs, ys, alpha=0.9)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Train Loss (step)")
        plt.savefig(os.path.join(out_dir, "train_loss_step.png"))
        plt.close(fig)

    # 按 epoch 收集
    epochs = sorted({entry["epoch"] for entry in log if "epoch" in entry})
    train_losses = []
    for ep in epochs:
        losses = [e["loss"] for e in log if "epoch" in e and abs(e["epoch"] - ep) < 1e-8 and "loss" in e]
        train_losses.append(losses[-1] if losses else None)

    # 收集 eval_* 指标
    metric_names = set()
    for e in log:
        for k in e.keys():
            if k.startswith("eval_"):
                metric_names.add(k[len("eval_"):])
    metric_names = sorted(metric_names)
    metrics = {name: [] for name in metric_names}
    for ep in epochs:
        entries = [e for e in log if "epoch" in e and abs(e["epoch"] - ep) < 1e-8]
        eval_entry = None
        for e in reversed(entries):
            if any(k.startswith("eval_") for k in e.keys()):
                eval_entry = e
                break
        for name in metric_names:
            metrics[name].append(eval_entry.get(f"eval_{name}") if eval_entry else None)

    # 保存按 epoch 的 train loss
    if any(v is not None for v in train_losses):
        sm_train_losses = smooth(train_losses)
        xs = [ep for ep, v in zip(epochs, sm_train_losses) if v is not None]
        ys = [v for v in sm_train_losses if v is not None]
        fig = plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("epoch")
        plt.ylabel("train_loss")
        plt.title("Train Loss (epoch)")
        plt.savefig(os.path.join(out_dir, "train_loss_epoch.png"))
        plt.close(fig)

    # 单独保存每个 eval 指标图，以及合并图
    for name, vals in metrics.items():
        if all(v is None for v in vals):
            continue
        sm_vals = smooth(vals)
        xs = [ep for ep, v in zip(epochs, sm_vals) if v is not None]
        ys = [v for v in sm_vals if v is not None]
        fig = plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.title(name)
        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close(fig)

    main_names = ["intent_accuracy", "slot_f1", "slot_precision", "slot_recall"]
    present = [n for n in main_names if n in metrics]
    if present:
        fig = plt.figure()
        for n in present:
            sm_n = smooth(metrics[n])
            xs = [ep for ep, v in zip(epochs, sm_n) if v is not None]
            ys = [v for v in sm_n if v is not None]
            plt.plot(xs, ys, marker="o", label=n)
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.legend()
        plt.title("Eval metrics")
        plt.savefig(os.path.join(out_dir, "eval_metrics.png"))
        plt.close(fig)

    print(f"训练图已保存至 {out_dir}")