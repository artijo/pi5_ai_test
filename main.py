#!/usr/bin/env python3
"""
Bus Passenger Counter - Main Application
Uses Pi5 + AI Kit (Hailo) + rpicam for counting passengers entering/exiting a bus

Camera is mounted on top looking down.
- Person moving from top to bottom = entering bus (IN)
- Person moving from bottom to top = exiting bus (OUT)
"""

import cv2
import json
import signal
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Import local modules
from src.config import *
from src.tracker import CentroidTracker, PassengerCounter
from src.utils import (
    save_results, draw_counting_line, draw_detection,
    draw_centroid, draw_trail, draw_counts, draw_event_notification,
    preprocess_frame, postprocess_detections
)

# Try to import Hailo runtime
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams
    HAILO_AVAILABLE = True
except ImportError:
    print("Warning: Hailo runtime not available. Using simulation mode.")
    HAILO_AVAILABLE = False

# Try to import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    print("Warning: picamera2 not available. Using OpenCV camera.")
    PICAMERA_AVAILABLE = False


class BusPassengerCounter:
    def __init__(self):
        """Initialize the bus passenger counter"""
        self.running = False
        self.start_time = None
        self.events = []
        
        # Initialize tracker
        self.tracker = CentroidTracker(
            max_disappeared=MAX_DISAPPEARED,
            max_distance=MAX_DISTANCE
        )
        
        # Initialize counter
        self.counter = PassengerCounter(
            line_position=COUNTING_LINE_POSITION,
            line_margin=COUNTING_LINE_MARGIN,
            frame_height=CAMERA_HEIGHT
        )
        
        # Initialize camera
        self.camera = None
        self.init_camera()
        
        # Initialize AI model
        self.model = None
        self.vdevice = None
        if HAILO_AVAILABLE:
            self.init_hailo_model()
        
        # Event notification state
        self.notification_event = None
        self.notification_frames = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def init_camera(self):
        """Initialize camera (Pi Camera or USB camera)"""
        self.use_picamera = False
        
        if PICAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.use_picamera = True
                print(f"Pi Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                return
            except Exception as e:
                print(f"Failed to initialize Pi Camera: {e}")
                self.camera = None
                
        # Fallback to OpenCV camera - try multiple indices
        print("Trying OpenCV camera...")
        for camera_index in [0, 1, 2, -1]:
            print(f"  Trying camera index {camera_index}...")
            self.camera = cv2.VideoCapture(camera_index)
            
            if self.camera is not None and self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                
                # Test read a frame
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    print(f"OpenCV Camera initialized on index {camera_index}: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                    return
                else:
                    self.camera.release()
                    
        print("Error: Could not open any camera!")
        print("Please check:")
        print("  1. Camera is connected properly")
        print("  2. Run 'rpicam-hello' to test Pi Camera")
        print("  3. Run 'ls /dev/video*' to check available cameras")
        self.camera = None
            
    def init_hailo_model(self):
        """Initialize Hailo AI model"""
        try:
            model_path = Path(HAILO_MODEL_PATH)
            if not model_path.exists():
                print(f"Warning: Model file not found: {HAILO_MODEL_PATH}")
                print("Available models in /usr/share/hailo-models/:")
                models_dir = Path("/usr/share/hailo-models/")
                if models_dir.exists():
                    for f in models_dir.glob("*.hef"):
                        print(f"  - {f}")
                return
                
            # Initialize Hailo device
            self.vdevice = VDevice()
            
            # Load HEF model
            hef = HEF(str(model_path))
            
            # Configure network
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.model = self.vdevice.configure(hef, configure_params)[0]
            
            print(f"Hailo model loaded: {HAILO_MODEL_PATH}")
            
        except Exception as e:
            print(f"Failed to initialize Hailo model: {e}")
            self.model = None
            
    def get_frame(self):
        """Get a frame from the camera"""
        if self.camera is None:
            return None
            
        try:
            if self.use_picamera and isinstance(self.camera, Picamera2):
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            else:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    return frame
        except Exception as e:
            print(f"Error getting frame: {e}")
            
        return None
        
    def detect_persons(self, frame):
        """
        Detect persons in frame using Hailo AI or fallback to simulation
        
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.model is not None:
            try:
                # Prepare input for Hailo
                input_data = self.preprocess_for_hailo(frame)
                
                # Run inference
                with self.model.activate():
                    # Get input/output virtual streams
                    input_vstreams = self.model.input_vstreams()
                    output_vstreams = self.model.output_vstreams()
                    
                    # Send data
                    for vs in input_vstreams:
                        vs.send(input_data)
                        
                    # Receive results
                    results = []
                    for vs in output_vstreams:
                        results.append(vs.recv())
                        
                # Post-process results
                boxes = self.postprocess_hailo_output(results, frame.shape)
                return boxes
                
            except Exception as e:
                print(f"Hailo inference error: {e}")
                return self.detect_persons_opencv(frame)
        else:
            # Fallback to OpenCV DNN or simulation
            return self.detect_persons_opencv(frame)
            
    def preprocess_for_hailo(self, frame):
        """Preprocess frame for Hailo inference"""
        # Resize to model input size (typically 640x640 for YOLO)
        input_size = (640, 640)
        resized = cv2.resize(frame, input_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension if needed
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
            
        return normalized
        
    def postprocess_hailo_output(self, results, original_shape):
        """Post-process Hailo model output"""
        boxes = []
        
        # Process based on model output format (YOLO format)
        for result in results:
            if result is None:
                continue
                
            # Typical YOLO output: [batch, num_detections, 6] 
            # where 6 = [x1, y1, x2, y2, confidence, class_id]
            detections = result.reshape(-1, result.shape[-1])
            
            for det in detections:
                if len(det) >= 5:
                    conf = det[4]
                    if conf >= CONFIDENCE_THRESHOLD:
                        # Get class (if available)
                        class_id = int(det[5]) if len(det) > 5 else 0
                        
                        if class_id == PERSON_CLASS_ID:
                            # Scale coordinates to original frame size
                            h, w = original_shape[:2]
                            x1 = int(det[0] * w / 640)
                            y1 = int(det[1] * h / 640)
                            x2 = int(det[2] * w / 640)
                            y2 = int(det[3] * h / 640)
                            boxes.append((x1, y1, x2, y2))
                            
        return boxes
        
    def detect_persons_opencv(self, frame):
        """
        Fallback person detection using OpenCV
        Uses HOG detector or pre-trained model
        """
        # Use HOG + SVM people detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people
        boxes, weights = hog.detectMultiScale(
            frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        # Convert to (x1, y1, x2, y2) format and filter by confidence
        result = []
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] >= CONFIDENCE_THRESHOLD:
                result.append((x, y, x + w, y + h))
                
        return result
        
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect persons
        boxes = self.detect_persons(frame)
        
        # Update tracker
        objects = self.tracker.update(boxes)
        
        # Update counter
        count_in, count_out, events = self.counter.update(objects, self.tracker.trails)
        
        # Store events with timestamp
        for event in events:
            event["timestamp"] = datetime.now().isoformat()
            self.events.append(event)
            self.notification_event = event["type"]
            self.notification_frames = 30
            
        return frame, boxes, objects, count_in, count_out
        
    def draw_visualization(self, frame, boxes, objects, count_in, count_out):
        """Draw visualization on frame"""
        # Draw counting line
        if DRAW_COUNTING_LINE:
            frame = draw_counting_line(frame, self.counter.line_y)
            
        # Draw bounding boxes and centroids
        for i, (object_id, centroid) in enumerate(objects.items()):
            if DRAW_BOUNDING_BOXES and i < len(boxes):
                frame = draw_detection(frame, boxes[i], object_id)
                
            if DRAW_CENTROIDS:
                frame = draw_centroid(frame, centroid, object_id)
                
            if DRAW_TRAILS:
                trail = self.tracker.get_trail(object_id, TRAIL_LENGTH)
                frame = draw_trail(frame, trail)
                
        # Draw counts
        frame = draw_counts(frame, count_in, count_out)
        
        # Draw event notification
        if self.notification_frames > 0 and self.notification_event:
            frame = draw_event_notification(frame, self.notification_event)
            self.notification_frames -= 1
            
        return frame
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down...")
        self.running = False
        
    def cleanup(self):
        """Clean up resources and save results"""
        end_time = datetime.now()
        
        # Save results to JSON (only if we have a start time)
        if self.start_time:
            counts = self.counter.get_counts()
            save_results(
                OUTPUT_JSON_FILE,
                counts["in"],
                counts["out"],
                self.start_time,
                end_time,
                self.events
            )
            
            # Print summary
            print("\n" + "="*50)
            print("SESSION SUMMARY")
            print("="*50)
            print(f"Duration: {end_time - self.start_time}")
            print(f"Total IN: {counts['in']}")
            print(f"Total OUT: {counts['out']}")
            print(f"Current in bus: {counts['current_in_bus']}")
            print(f"Results saved to: {OUTPUT_JSON_FILE}")
            print("="*50)
        
        # Close camera
        try:
            if self.use_picamera and isinstance(self.camera, Picamera2):
                self.camera.stop()
                self.camera.close()
            elif self.camera is not None:
                self.camera.release()
        except Exception as e:
            print(f"Error closing camera: {e}")
            
        # Close display window
        cv2.destroyAllWindows()
        
    def run(self):
        """Main loop"""
        self.running = True
        self.start_time = datetime.now()
        
        print("\n" + "="*50)
        print("BUS PASSENGER COUNTER")
        print("="*50)
        print(f"Started at: {self.start_time}")
        print("Press 'q' to quit and save results")
        print("Press 'r' to reset counts")
        print("="*50 + "\n")
        
        # Check if camera is available
        if self.camera is None:
            print("Error: No camera available! Exiting...")
            self.cleanup()
            return
        
        # Start camera if Pi Camera
        if self.use_picamera and isinstance(self.camera, Picamera2):
            self.camera.start()
            time.sleep(0.5)  # Wait for camera to warm up
            
        frame_error_count = 0
        max_frame_errors = 50  # Stop after 50 consecutive errors
            
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while self.running:
            # Get frame
            frame = self.get_frame()
            if frame is None:
                frame_error_count += 1
                if frame_error_count >= max_frame_errors:
                    print(f"Error: Too many frame errors ({frame_error_count}). Stopping...")
                    break
                if frame_error_count % 10 == 1:
                    print(f"Warning: Could not get frame (error #{frame_error_count})")
                time.sleep(0.1)
                continue
            
            # Reset error count on successful frame
            frame_error_count = 0
                
            # Process frame
            frame, boxes, objects, count_in, count_out = self.process_frame(frame)
            
            # Draw visualization
            if SHOW_PREVIEW:
                frame = self.draw_visualization(frame, boxes, objects, count_in, count_out)
                
                # Calculate and display FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                    
                cv2.putText(frame, f"FPS: {current_fps}", (CAMERA_WIDTH - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow("Bus Passenger Counter", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    self.counter.reset()
                    self.events = []
                    print("Counts reset!")
                    
        # Cleanup and save results
        self.cleanup()


def main():
    """Entry point"""
    counter = BusPassengerCounter()
    counter.run()


if __name__ == "__main__":
    main()
