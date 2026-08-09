import torch
import torch.nn as nn

class PCELoss(nn.Module):
    """
    Partial Cross Entropy Loss 
        - evaluates only at labeled points.
    """

    def __init__(self, ignore_index=-1, class_weights=None):
        super().__init__()

        self.ignore_index = ignore_index
        if class_weights is not None:
            self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, point_labels):
        if logits.ndim != 4 or point_labels.ndim != 3:
            raise ValueError("Excepted logints [B, C, H, W] and labels [B, H, W]")

        return nn.functional.cross_entropy(logits,
                                           point_labels.long(),
                                           weight=self.class_weights,
                                           ignore_index=self.ignore_index,
                                           )
        
