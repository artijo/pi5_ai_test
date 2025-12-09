"""
Centroid Tracker for tracking people across frames
Uses centroid-based tracking with direction detection
"""

import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Initialize the centroid tracker
        
        Args:
            max_disappeared: Maximum consecutive frames an object can be missing
            max_distance: Maximum distance between centroids for matching
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # {object_id: centroid}
        self.disappeared = OrderedDict()  # {object_id: frames_missing}
        self.trails = OrderedDict()  # {object_id: list of centroids}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        """Register a new object with the next available ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trails[self.next_object_id] = [centroid]
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Deregister an object ID"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trails[object_id]
        
    def update(self, rects):
        """
        Update tracker with new detections
        
        Args:
            rects: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Dictionary of {object_id: centroid}
        """
        # If no detections, mark all existing objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        # Compute centroids for current detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            
        # If no existing objects, register all new centroids
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing objects with new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance between each pair of existing and new centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance for each row, then sort
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                # If distance is too large, skip
                if D[row, col] > self.max_distance:
                    continue
                    
                # Update the object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.trails[object_id].append(input_centroids[col])
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            # Handle unmatched existing objects
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            
            # If we have more existing objects than detections
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # Register new objects
                for col in unused_cols:
                    self.register(input_centroids[col])
                    
        return self.objects
    
    def get_trail(self, object_id, max_length=30):
        """Get the trail of an object"""
        if object_id in self.trails:
            return self.trails[object_id][-max_length:]
        return []


class PassengerCounter:
    def __init__(self, line_position, line_margin=20, frame_height=480):
        """
        Initialize the passenger counter
        
        Args:
            line_position: Y coordinate of the counting line (or percentage 0-1)
            line_margin: Margin around the line for detection
            frame_height: Height of the frame
        """
        if 0 < line_position <= 1:
            self.line_y = int(line_position * frame_height)
        else:
            self.line_y = int(line_position)
            
        self.line_margin = line_margin
        self.frame_height = frame_height
        
        # Counting results
        self.count_in = 0
        self.count_out = 0
        
        # Track which objects have been counted
        self.counted_objects = set()
        
        # Previous positions for direction detection
        self.previous_positions = {}
        
    def update(self, objects, trails):
        """
        Update counter based on tracked objects
        
        Args:
            objects: Dictionary of {object_id: centroid}
            trails: Dictionary of {object_id: list of centroids}
            
        Returns:
            Tuple of (count_in, count_out, events)
        """
        events = []
        
        for object_id, centroid in objects.items():
            # Skip if already counted
            if object_id in self.counted_objects:
                continue
                
            cx, cy = centroid
            
            # Check if object crossed the counting line
            if object_id in self.previous_positions:
                prev_y = self.previous_positions[object_id]
                
                # Check if crossed from top to bottom (IN)
                if prev_y < self.line_y and cy >= self.line_y:
                    self.count_in += 1
                    self.counted_objects.add(object_id)
                    events.append({
                        "type": "in",
                        "object_id": object_id,
                        "position": (cx, cy)
                    })
                    
                # Check if crossed from bottom to top (OUT)
                elif prev_y > self.line_y and cy <= self.line_y:
                    self.count_out += 1
                    self.counted_objects.add(object_id)
                    events.append({
                        "type": "out",
                        "object_id": object_id,
                        "position": (cx, cy)
                    })
                    
            # Update previous position
            self.previous_positions[object_id] = cy
            
        # Clean up old objects
        current_ids = set(objects.keys())
        for obj_id in list(self.previous_positions.keys()):
            if obj_id not in current_ids:
                del self.previous_positions[obj_id]
                
        return self.count_in, self.count_out, events
    
    def get_counts(self):
        """Get current counts"""
        return {
            "in": self.count_in,
            "out": self.count_out,
            "current_in_bus": self.count_in - self.count_out
        }
    
    def reset(self):
        """Reset all counts"""
        self.count_in = 0
        self.count_out = 0
        self.counted_objects.clear()
        self.previous_positions.clear()
