from torch.utils.data import Dataset
import torch
import os
from PIL import Image

class YOLODataset(Dataset):
    """
    Custom dataset for YOLO object detection.

    This dataset loads images and their corresponding labels,
    applies necessary transformations, and prepares target tensors
    for training a YOLO model.
    """
    def __init__(self, image_dir, label_dir, anchors, grid_size=13, num_classes=4, transform=None):
        """
        Initializes the dataset.
        
        Args:
            image_dir (str): Path to the directory containing images.
            label_dir (str): Path to the directory containing labels.
            anchors (list of tuples): Predefined anchor box sizes.
            grid_size (int): Size of the grid (e.g., 13x13 for YOLOv2).
            num_classes (int): Number of object classes.
            transform (callable, optional): Image transformations.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors)  # Shape: [num_anchors, 2]
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads an image and its corresponding label, and prepares the target tensor.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            image (Tensor): Transformed image.
            target (Tensor): YOLO-formatted target tensor of shape.
        """
        # Construct file paths for image and label
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Parse label data
        objects = []
        for line in lines:
            class_id, cx, cy, w, h = map(float, line.strip().split())
            objects.append({
                'class_id': int(class_id),
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h
            })
        
        # Initialize target tensor
        num_anchors = len(self.anchors)
        target = torch.zeros((num_anchors, self.grid_size, self.grid_size, 5 + self.num_classes))
        
        for obj in objects:
            cx, cy, w, h, class_id = obj['cx'], obj['cy'], obj['w'], obj['h'], obj['class_id']
            
            # Grid cell indices
            i = int(cx * self.grid_size)
            j = int(cy * self.grid_size)
            i = max(0, min(i, self.grid_size - 1))
            j = max(0, min(j, self.grid_size - 1))
            
            # Compute IoU with each anchor
            gt_box = torch.tensor([w, h])
            ious = []
            for anchor in self.anchors:
                inter_w = min(w, anchor[0])
                inter_h = min(h, anchor[1])
                inter_area = inter_w * inter_h
                union_area = w * h + anchor[0] * anchor[1] - inter_area
                iou = inter_area / union_area
                ious.append(iou)

            # Select the best anchor (highest IoU)
            best_anchor = torch.argmax(torch.tensor(ious)).item()
            
            # Compute target values
            tx = cx * self.grid_size - i
            ty = cy * self.grid_size - j
            tw = torch.log(torch.tensor(w / self.anchors[best_anchor][0]))
            th = torch.log(torch.tensor(h / self.anchors[best_anchor][1]))
            one_hot = torch.zeros(self.num_classes)
            one_hot[class_id] = 1.0
            
            # Assign to target tensor
            target[best_anchor, j, i, 0] = tx
            target[best_anchor, j, i, 1] = ty
            target[best_anchor, j, i, 2] = tw
            target[best_anchor, j, i, 3] = th
            target[best_anchor, j, i, 4] = 1.0  # Objectness
            target[best_anchor, j, i, 5:] = one_hot
        
        return image, target