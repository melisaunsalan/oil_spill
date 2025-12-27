import torch
import torch.nn.functional as F

def confusion_weighted_ce(
    logits,
    targets,
    class_a=1,
    class_b=2,
    penalty=3.0,
    ignore_index=255
):
    """
    logits: [B, C, H, W]
    targets: [B, H, W]
    """

    # Standard CE per pixel
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        ignore_index=ignore_index
    )

    with torch.no_grad():
        preds = logits.argmax(dim=1)

        confusion_mask = (
            ((targets == class_a) & (preds == class_b)) |
            ((targets == class_b) & (preds == class_a))
        )

    weights = torch.ones_like(ce)
    weights[confusion_mask] *= penalty

    return (ce * weights).mean()
