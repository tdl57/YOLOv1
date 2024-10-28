import torch
from iou import intersection_over_union

def non_max_supression(bboxes, iou_threshold, prob_threshold, box_format='corners'):
    # bboxes = [[1, 0.9, x1, y1, x2, y2], [], [], etc. etc.]
    # each box is in format[class, confidence, x1, y1, x2, y2]

    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_bbox = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_bbox[0] 
            or intersection_over_union(torch.tensor(chosen_bbox[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_bbox)
    
    return bboxes_after_nms

    