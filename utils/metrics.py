import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score

def evaluate_model_with_classacc(model, data, edge_index, edge_weight=None, num_classes=None):
    """
    评估模型在 train/val/test 三个划分上的性能，
    同时计算每个类别的准确率。

    返回：
    {
        'train_mask': {'accuracy':..., 'macro_f1':..., 'bacc':..., 'class_acc':{0:...,1:...}},
        'val_mask': {...},
        'test_mask': {...}
    }
    """
    import torch
    from sklearn.metrics import f1_score, balanced_accuracy_score

    model.eval()
    with torch.no_grad():
        # === 前向传播 ===
        if edge_weight is not None:
            out = model(data.x, data.edge_index, edge_weight=edge_weight)
        else:
            out = model(data.x, data.edge_index)

        logits = out
        preds = logits.argmax(dim=1)
        y_true = data.y

        if num_classes is None:
            num_classes = int(y_true.max().item()) + 1

        results = {}

        # === 针对 train/val/test 三个划分分别计算 ===
        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = getattr(data, split)
            mask_np = mask.cpu().numpy()

            pred_split = preds[mask].cpu().numpy()
            true_split = y_true[mask].cpu().numpy()

            acc = (pred_split == true_split).mean()
            f1 = f1_score(true_split, pred_split, average='macro')
            bacc = balanced_accuracy_score(true_split, pred_split)

            # === 类别准确率 ===
            class_acc = {}
            for c in range(num_classes):
                idx = (true_split == c)
                if idx.sum() == 0:
                    class_acc[c] = float('nan')
                else:
                    correct = (pred_split[idx] == c).sum()
                    class_acc[c] = correct / idx.sum()

            results[split] = {
                'accuracy': acc,
                'macro_f1': f1,
                'bacc': bacc,
                'class_acc': class_acc
            }

    return results

