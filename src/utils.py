"""
Utility functions for Bus Passenger Counter
"""

import json
import cv2
import numpy as np
from datetime import datetime


def save_results(filename, count_in, count_out, start_time, end_time, events=None):
    """
    Save counting results to JSON file
    
    Args:
        filename: Output file path
        count_in: Total passengers entered
        count_out: Total passengers exited
        start_time: Session start time
        end_time: Session end time
        events: List of counting events (optional)
    """
    result = {
        "session": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        },
        "counts": {
            "total_in": count_in,
            "total_out": count_out,
            "current_in_bus": count_in - count_out
        },
        "summary": {
            "net_change": count_in - count_out,
            "total_movements": count_in + count_out
        }
    }
    
    if events:
        result["events"] = events
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {filename}")
    return result


def draw_counting_line(frame, line_y, color=(0, 255, 255), thickness=2):
    """Draw the counting line on frame"""
    height, width = frame.shape[:2]
    cv2.line(frame, (0, line_y), (width, line_y), color, thickness)
    return frame


def draw_detection(frame, bbox, object_id, color=(0, 255, 0), thickness=2):
    """Draw bounding box and object ID"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw object ID
    label = f"ID: {object_id}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def draw_centroid(frame, centroid, object_id, color=(255, 0, 0), radius=5):
    """Draw centroid point"""
    cx, cy = [int(v) for v in centroid]
    cv2.circle(frame, (cx, cy), radius, color, -1)
    return frame


def draw_trail(frame, trail, color=(255, 255, 0), thickness=2):
    """Draw object trail"""
    if len(trail) < 2:
        return frame
        
    for i in range(1, len(trail)):
        pt1 = tuple([int(v) for v in trail[i-1]])
        pt2 = tuple([int(v) for v in trail[i]])
        # Fade color based on age
        alpha = i / len(trail)
        line_color = tuple([int(c * alpha) for c in color])
        cv2.line(frame, pt1, pt2, line_color, thickness)
        
    return frame


def draw_counts(frame, count_in, count_out, position=(10, 30)):
    """Draw counting information on frame"""
    current = count_in - count_out
    
    # Background rectangle
    cv2.rectangle(frame, (5, 5), (200, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (200, 100), (255, 255, 255), 1)
    
    # Count text
    cv2.putText(frame, f"IN: {count_in}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {count_out}", (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Current: {current}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame


def draw_event_notification(frame, event_type, duration_frames=30):
    """Draw event notification (IN/OUT)"""
    height, width = frame.shape[:2]
    
    if event_type == "in":
        color = (0, 255, 0)
        text = "PASSENGER IN"
    else:
        color = (0, 0, 255)
        text = "PASSENGER OUT"
        
    # Draw notification
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x = (width - text_width) // 2
    y = height - 50
    
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def preprocess_frame(frame, target_size=(640, 480)):
    """Preprocess frame for detection"""
    if frame is None:
        return None
    
    # Resize if needed
    if frame.shape[:2] != target_size[::-1]:
        frame = cv2.resize(frame, target_size)
        
    return frame


def postprocess_detections(detections, confidence_threshold=0.5, person_class_id=0):
    """
    Filter detections to only include persons above confidence threshold
    
    Args:
        detections: Raw detections from model
        confidence_threshold: Minimum confidence
        person_class_id: Class ID for person (usually 0 in COCO)
        
    Returns:
        List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    boxes = []
    
    for detection in detections:
        # Assuming detection format: [x1, y1, x2, y2, confidence, class_id]
        # Adjust based on actual model output
        if len(detection) >= 6:
            x1, y1, x2, y2, conf, class_id = detection[:6]
        elif len(detection) >= 5:
            x1, y1, x2, y2, conf = detection[:5]
            class_id = person_class_id  # Assume person if no class
        else:
            continue
            
        if conf >= confidence_threshold and int(class_id) == person_class_id:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
    return boxes


def apply_nms(boxes, scores, nms_threshold=0.4):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
        
    boxes_array = np.array(boxes)
    scores_array = np.array(scores)
    
    indices = cv2.dnn.NMSBoxes(
        boxes_array.tolist(),
        scores_array.tolist(),
        score_threshold=0.0,
        nms_threshold=nms_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices]
    
    return []
