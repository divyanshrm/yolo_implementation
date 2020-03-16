def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    Converts the output of YOLO encoding (a lot of boxes) to predicted boxes along with their scores, box coordinates and classes.
    
     
    
    
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    
    scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores,boxes,classes,iou_threshold=iou_threshold)
    
    
    
    return scores, boxes, classes