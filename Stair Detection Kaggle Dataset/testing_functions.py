import torch

import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

def forward_pass(model, images, targets=None):
    """
    Perform a forward pass through the model.

    Args:
        model (nn.Module): The object detection model.
        images (torch.Tensor): Input images of shape [batch_size, channels, height, width].
        targets (dict): Ground truth with keys 'boxes' and 'labels'.
    
    Returns:
        Tuple: Predicted bounding boxes and class logits (during inference),
               or total loss (during training).
    """
    # Perform forward pass through the model
    bbox_predictions, class_logits = model(images)
    
    if targets:
        # If training, compute losses
        loss = compute_losses(bbox_predictions, class_logits, targets)
        return loss
    else:
        # If inference, return predictions
        return bbox_predictions, class_logits

def compute_losses(bbox_predictions, class_logits, targets):
    """
    Compute the total loss for bounding box regression and classification.

    Args:
        bbox_predictions (torch.Tensor): Predicted bounding boxes, shape [batch_size, 4].
        class_logits (torch.Tensor): Predicted class logits, shape [batch_size, num_classes].
        targets (dict): Ground truth with keys:
            - 'boxes': Ground truth bounding boxes, shape [batch_size, 4].
            - 'labels': Ground truth class labels, shape [batch_size].

    Returns:
        torch.Tensor: Total loss (sum of regression and classification losses).
    """
    # Smooth L1 Loss for bounding box regression
    bbox_loss_fn = nn.SmoothL1Loss()
    bbox_loss = bbox_loss_fn(bbox_predictions, targets['boxes'])
    
    # Cross-Entropy Loss for classification
    class_loss_fn = nn.CrossEntropyLoss()
    class_loss = class_loss_fn(class_logits, targets['labels'])
    
    # Total loss
    total_loss = bbox_loss + class_loss
    return total_loss


def generate_anchors(base_size=16, scales=[0.5, 1.0, 2.0], aspect_ratios=[0.5, 1.0, 2.0]):
    """
    Generate anchor boxes based on scales and aspect ratios.

    Args:
        base_size (int): The size of the base anchor.
        scales (list): Scaling factors for anchors.
        aspect_ratios (list): Aspect ratios for anchors.

    Returns:
        torch.Tensor: Generated anchors of shape [num_anchors, 4] (x_min, y_min, x_max, y_max).
    """
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            x_min, y_min = -w / 2, -h / 2
            x_max, y_max = w / 2, h / 2
            anchors.append([x_min, y_min, x_max, y_max])
    
    return torch.tensor(anchors, dtype=torch.float32)




def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform non-maximum suppression (NMS) on bounding boxes.

    Args:
        boxes (torch.Tensor): Predicted boxes, shape [num_boxes, 4].
        scores (torch.Tensor): Confidence scores, shape [num_boxes].
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        torch.Tensor: Indices of the retained boxes.
    """
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while indices.numel() > 0:
        current = indices[0]
        keep.append(current)
        
        if indices.numel() == 1:
            break
        
        remaining_boxes = boxes[indices[1:]]
        iou = calculate_iou(boxes[current].unsqueeze(0), remaining_boxes)
        indices = indices[1:][iou < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long)

def calculate_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1 (torch.Tensor): Single box, shape [1, 4].
        box2 (torch.Tensor): Multiple boxes, shape [N, 4].
    
    Returns:
        torch.Tensor: IoU scores for each box in box2.
    """
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]))
    

    area1= (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2= (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    x1= torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1= torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2= torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2= torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    inter_w= torch.clamp(x2 - x1, min=0)
    inter_h= torch.clamp(y2 - y1, min=0)
    inter_area= inter_w * inter_h
    union_area= area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    iou= inter_area / (union_area + 1e-6)


    return iou  # shape [N, M]

def evaluate_model(model, dataloader, iou_threshold=0.5):
    """Evaluate mAP for the object detection model."""
    model.eval()
    all_precisions = []
    all_recalls = []

    for batch in dataloader:
        images = batch['image']
        true_boxes = batch['boxes']
        true_labels = batch['labels']

        with torch.no_grad():
            predictions = model(images)
        
        for i, pred in enumerate(predictions):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = true_boxes[i]
            
            # Calculate IoU for each prediction
            matches = []
            for pred_box in pred_boxes:
                ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
                matches.append(max(ious) >= iou_threshold)
            
            tp = sum(matches)
            fp = len(matches) - tp
            fn = len(gt_boxes) - tp
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)

    mAP = np.mean(all_precisions)
    return mAP
    

def plot_dataset_item(dataset, idx):
    image, target, image_name = dataset[idx]
    image_to_plot = image.permute(1, 2, 0).cpu().numpy() # Convert to HWC format for plotting
    plt.imshow(image_to_plot)
    plt.title(f"Image: {image_name}")
    
    boxes = target['boxes'].numpy()
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    plt.show()