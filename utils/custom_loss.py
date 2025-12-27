import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerCustomLoss(nn.Module):
    def __init__(self, penalty_weight=5.0, class_1 = 1, class_2 = 2):
        super(SegFormerCustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.class_1 = class_1
        self.class_2 = class_2
        # Use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
        # It expects raw scores (logits) as input.
        self.base_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, y_pred, y_true):
        # y_pred: (batch_size, num_classes, H, W)
        # y_true: (batch_size, H, W) with class indices

        # Ensure y_true is long type for CrossEntropyLoss
        y_true = y_true.long()

        # Calculate base CrossEntropyLoss
        # nn.CrossEntropyLoss expects y_pred as (N, C, ...) and y_true as (N, ...)
        base_loss = self.base_criterion(y_pred, y_true)

        # --- Penalty for confusion between class 1 and class 2 ---

        # Get predicted class indices (highest probability)
        # y_pred_labels will be (batch_size, H, W)
        y_pred_labels = torch.argmax(y_pred, dim=1).long()

        # Create masks for true labels being class 1 or class 2
        true_class_1_mask = (y_true == self.class_1).float()
        true_class_2_mask = (y_true == self.class2).float()

        # Create masks for predicted labels being class 1 or class 2
        pred_class_1_mask = (y_pred_labels == self.class_1).float()
        pred_class_2_mask = (y_pred_labels == self.class_2).float()

        # Confusion: true is 1, pred is 2
        confusion_1_to_2 = true_class_1_mask * pred_class_2_mask

        # Confusion: true is 2, pred is 1
        confusion_2_to_1 = true_class_2_mask * pred_class_1_mask

        # Sum of confusion across all pixels in the batch
        # Normalize by the total number of pixels to get an average confusion value per image
        # And then average across the batch for consistency with base_loss reduction.
        
        # Calculate number of pixels per image
        num_pixels = y_true.shape[1] * y_true.shape[2]
        
        total_confusion_per_image = torch.sum(confusion_1_to_2 + confusion_2_to_1, dim=(1, 2))
        avg_confusion_over_batch = torch.mean(total_confusion_per_image / num_pixels)

        penalty_term = avg_confusion_over_batch * self.penalty_weight

        return base_loss + penalty_term

