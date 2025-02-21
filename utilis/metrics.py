import torch
from torchmetrics.detection import MeanAveragePrecision

class YOLOMetrics:
    """
    Computes evaluation metrics for YOLO object detection. It calculates different mean Average Precision (mAP) scores.
    """
    def __init__(self, num_classes, anchors, grid_size=13, conf_threshold=0.5, iou_threshold=0.5):
        """
        Initializes the metrics.

        Args:
            num_classes (int): Number of object classes.
            anchors (list): List of anchor box dimensions.
            grid_size (int): Size of the output grid (default: 13).
            conf_threshold (float): Confidence threshold for filtering predictions (default: 0.5).
            iou_threshold (float): IoU threshold for metric calculation (default: 0.5).
        """
        self.num_classes = num_classes
        self.anchors = anchors
        self.grid_size = grid_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    def _decode_preds(self, predictions):
        """
        Converts model predictions into bounding boxes, confidence scores, and class probabilities.

        Args:
            predictions (torch.Tensor): Model output.

        Returns:
            Tuple of tensors containing bounding box coordinates (bx, by, bw, bh), objectness scores, and class probabilities.
        """
        batch_size, _, grid_h, grid_w = predictions.size()
        num_anchors = len(self.anchors)
        
        # Reshape predictions
        pred = predictions.view(batch_size, num_anchors, 5 + self.num_classes, grid_h, grid_w)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Get box coordinates
        tx = torch.sigmoid(pred[..., 0])
        ty = torch.sigmoid(pred[..., 1])
        tw = pred[..., 2]
        th = pred[..., 3]
        obj_score = torch.sigmoid(pred[..., 4])
        class_probs = torch.softmax(pred[..., 5:], dim=-1)
        
        # Convert to actual coordinates
        grid_x = torch.arange(grid_w).repeat(grid_h, 1).view([1, 1, grid_h, grid_w]).float()
        grid_y = torch.arange(grid_h).repeat(grid_w, 1).t().view([1, 1, grid_h, grid_w]).float()
        
        anchor_w = torch.tensor(self.anchors)[:, 0].view(1, num_anchors, 1, 1)
        anchor_h = torch.tensor(self.anchors)[:, 1].view(1, num_anchors, 1, 1)
        
        bx = (tx + grid_x) / grid_w
        by = (ty + grid_y) / grid_h
        bw = anchor_w * torch.exp(tw)
        bh = anchor_h * torch.exp(th)
        
        return bx, by, bw, bh, obj_score, class_probs

    def update_metrics(self, predictions, targets):
        """
        Update metrics for current batch.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            None
        """
        bx, by, bw, bh, obj_score, class_probs = self._decode_preds(predictions)
        batch_size = predictions.size(0)
        
        detections = []
        ground_truths = []
        
        for batch_idx in range(batch_size):
            # Process predictions
            scores = obj_score[batch_idx]
            class_ids = torch.argmax(class_probs[batch_idx], dim=-1)
            
            # Filter by confidence
            mask = scores > self.conf_threshold
            boxes = torch.stack([
                bx[batch_idx][mask] - bw[batch_idx][mask]/2,
                by[batch_idx][mask] - bh[batch_idx][mask]/2,
                bx[batch_idx][mask] + bw[batch_idx][mask]/2,
                by[batch_idx][mask] + bh[batch_idx][mask]/2,
            ], dim=-1)
            
            dets = {
                'boxes': boxes,
                'scores': scores[mask],
                'labels': class_ids[mask]
            }
            detections.append(dets)
            
            # Process ground truth
            gt = targets[batch_idx]
            gt_boxes = []
            gt_labels = []
            
            for anchor_idx in range(len(self.anchors)):
                anchor_target = gt[anchor_idx]  # Shape [grid_h, grid_w, 5+num_classes]
                obj_mask = anchor_target[..., 4] == 1
                grid_h, grid_w = anchor_target.shape[:2]
                
                # Get indices where there's an object
                j_indices, i_indices = torch.where(obj_mask)
                
                for j, i in zip(j_indices, i_indices):
                    tx = anchor_target[j, i, 0]
                    ty = anchor_target[j, i, 1]
                    tw = anchor_target[j, i, 2]
                    th = anchor_target[j, i, 3]
                    class_id = torch.argmax(anchor_target[j, i, 5:])
                    
                    # Decode ground truth box
                    cx = (i + tx) / self.grid_size
                    cy = (j + ty) / self.grid_size
                    anchor_w = self.anchors[anchor_idx][0]
                    anchor_h = self.anchors[anchor_idx][1]
                    w = anchor_w * torch.exp(tw)
                    h = anchor_h * torch.exp(th)
                    
                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2
                    
                    gt_boxes.append(torch.tensor([x_min, y_min, x_max, y_max]))
                    gt_labels.append(class_id)
            
            if gt_boxes:
                gt_boxes_tensor = torch.stack(gt_boxes)
                gt_labels_tensor = torch.stack(gt_labels)
            else:
                gt_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels_tensor = torch.zeros(0, dtype=torch.int64)
            
            ground_truths.append({
                'boxes': gt_boxes_tensor,
                'labels': gt_labels_tensor
            })
            
        # Update metric with corrected ground truths
        self.map_metric.update(detections, ground_truths)

    def compute_metrics(self):
        """
        Compute final evaluation metrics

        Returns:
            dict: Dictionary containing mean Average Precision (mAP) scores at different IoU thresholds.
        """
        map_results = self.map_metric.compute()
        return {
            'map': map_results['map'],
            'map_50': map_results['map_50'],
            'map_75': map_results['map_75'],
            'class_accuracy': map_results['map_per_class']
        }

    def reset(self):
        """
        Resets the metrics, clearing any accumulated data.

        Returns:
            None
        """
        self.map_metric.reset()