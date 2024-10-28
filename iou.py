import torch 

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    """
    Calculates the Intersection over Union (IoU) between two sets of boxes.
    
    Parameters:
    boxes_preds: [N, 4] where N is the number of boxes
    boxes_labels: [N, 4] where N is the number of boxes
    box_format: the format of the boxes. Can be 'midpoint' or 'corners'

    Returns:
        iou 
    """
    
    if box_format == "midpoint":
        box1_x1 = boxes_preds[:, 0:1] - boxes_preds[:, 2:3] / 2
        box1_y1 = boxes_preds[:, 1:2] - boxes_preds[:, 3:4] / 2
        box1_x2 = boxes_preds[:, 0:1] + boxes_preds[:, 2:3] / 2
        box1_y2 = boxes_preds[:, 1:2] + boxes_preds[:, 3:4] / 2
    
        box2_x1 = boxes_labels[:, 0:1] - boxes_labels[:, 2:3] / 2
        box2_y1 = boxes_labels[:, 1:2] - boxes_labels[:, 3:4] / 2
        box2_x2 = boxes_labels[:, 0:1] + boxes_labels[:, 2:3] / 2
        box2_y2 = boxes_labels[:, 1:2] + boxes_labels[:, 3:4] / 2
    
        x1_max = torch.max(box1_x1, box2_x1)
        y1_max = torch.max(box1_y1, box2_y1)
        x2_min = torch.min(box1_x2, box2_x2)
        y2_min = torch.min(box1_y2, box2_y2)
    
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        union = (box1_x2 - box1_x1) * (box1_y2 - box1_y1) + (box2_x2 - box2_x1) * (box2_y2 - box2_y1) - intersection
        
    if box_format == "corners":
        box1_x1 = boxes_preds[:, 0:1]
        box1_y1 = boxes_preds[:, 1:2]
        box1_x2 = boxes_preds[:, 2:3]
        box1_y2 = boxes_preds[:, 3:4]
    
        box2_x1 = boxes_labels[:, 0:1]
        box2_y1 = boxes_labels[:, 1:2]
        box2_x2 = boxes_labels[:, 2:3]
        box2_y2 = boxes_labels[:, 3:4]
    
        x1_max = torch.max(box1_x1, box2_x1)
        y1_max = torch.max(box1_y1, box2_y1)
        x2_min = torch.min(box1_x2, box2_x2)
        y2_min = torch.min(box1_y2, box2_y2)
    
    # .clamp(min=0) for when there is no intersection
    intersection = (x2_min - x1_max).clamp(0) * (y2_min - y1_max).clamp(0)
    union = (box1_x2 - box1_x1) * (box1_y2 - box1_y1) + (box2_x2 - box2_x1) * (box2_y2 - box2_y1) - intersection
    
    return (intersection / union) + 1e-6 