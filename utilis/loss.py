import torch.nn as nn
import torch

class YOLOLoss(nn.Module):
    """
    Computes the custom loss for YOLO object detection.
    """
    def __init__(self, num_classes, anchors, lambda_coord=5, lambda_noobj=0.5):
        """
        Initializes the loss.

        Args:
            num_classes (int): Number of object classes.
            anchors (list): List of anchor box dimensions.
            lambda_coord (float): Weight for coordinate loss (default: 5).
            lambda_noobj (float): Weight for no-object confidence loss (default: 0.5).
        """
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, predictions, targets):
        """
        Computes YOLO loss components: localization, confidence, and classification.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.

        Returns:
            torch.Tensor: Total loss normalized by batch size.
        """
        batch_size, _, grid_h, grid_w = predictions.size()
        num_anchors = len(self.anchors)
        predictions = predictions.view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        
        tx_pred = torch.sigmoid(predictions[..., 0])
        ty_pred = torch.sigmoid(predictions[..., 1])
        tw_pred = predictions[..., 2]
        th_pred = predictions[..., 3]
        # Extract objectness and class predictions
        obj_score = predictions[..., 4]
        class_scores = predictions[..., 5:]
        
        target_tx = targets[..., 0]
        target_ty = targets[..., 1]
        target_tw = targets[..., 2]
        target_th = targets[..., 3]
        # Extract ground truth labels
        target_obj = targets[..., 4]
        target_class = targets[..., 5:]

        # Object and No-Object masks
        obj_mask = target_obj == 1
        noobj_mask = target_obj == 0
        
        # Localization Loss
        loss_x = self.mse(tx_pred[obj_mask], target_tx[obj_mask])
        loss_y = self.mse(ty_pred[obj_mask], target_ty[obj_mask])
        loss_w = self.mse(tw_pred[obj_mask], target_tw[obj_mask])
        loss_h = self.mse(th_pred[obj_mask], target_th[obj_mask])
        loc_loss = (loss_x + loss_y + loss_w + loss_h) * self.lambda_coord
        
        # Confidence Loss
        conf_loss_obj = self.bce(obj_score[obj_mask], target_obj[obj_mask])
        conf_loss_noobj = self.bce(obj_score[noobj_mask], target_obj[noobj_mask]) * self.lambda_noobj
        conf_loss = conf_loss_obj + conf_loss_noobj
        
        # Classification Loss
        _, target_class_idx = target_class[obj_mask].max(dim=1)
        class_loss = self.ce(class_scores[obj_mask], target_class_idx)

        total_loss = (loc_loss + conf_loss + class_loss) / batch_size
        return total_loss