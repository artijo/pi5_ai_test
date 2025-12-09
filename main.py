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
import subprocess
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
    print("picamera2 is available")
except ImportError:
    print("Warning: picamera2 not available. Will try libcamera/OpenCV.")
    PICAMERA_AVAILABLE = False


class LibcameraCapture:
    """
    Capture frames using rpicam-vid (Pi5) or libcamera-vid and pipe to OpenCV
    This works when rpicam-hello works but picamera2 is not installed
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.running = False
        
    def start(self):
        """Start the rpicam/libcamera capture process"""
        # Try rpicam-vid first (Pi5 with Bookworm), then libcamera-vid
        commands_to_try = [
            ['rpicam-vid', '--inline', '--nopreview', '-t', '0',
             '--width', str(self.width), '--height', str(self.height),
             '--framerate', str(self.fps), '--codec', 'mjpeg', '-o', '-'],
            ['libcamera-vid', '--inline', '--nopreview', '-t', '0',
             '--width', str(self.width), '--height', str(self.height),
             '--framerate', str(self.fps), '--codec', 'mjpeg', '-o', '-'],
        ]
        
        for cmd in commands_to_try:
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8
                )
                self.running = True
                print(f"{cmd[0]} started: {self.width}x{self.height}@{self.fps}fps")
                return True
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Failed to start {cmd[0]}: {e}")
                continue
                
        return False
            
    def read(self):
        """Read a frame from the libcamera stream"""
        if not self.running or self.process is None:
            return False, None
            
        try:
            # Read JPEG data from pipe
            # JPEG starts with 0xFFD8 and ends with 0xFFD9
            jpeg_data = b''
            while True:
                byte = self.process.stdout.read(1)
                if not byte:
                    return False, None
                jpeg_data += byte
                
                # Check for JPEG start marker
                if len(jpeg_data) >= 2 and jpeg_data[-2:] == b'\xff\xd8':
                    jpeg_data = b'\xff\xd8'
                    
                # Check for JPEG end marker
                if len(jpeg_data) > 2 and jpeg_data[-2:] == b'\xff\xd9':
                    break
                    
                # Safety limit
                if len(jpeg_data) > 10**7:
                    return False, None
                    
            # Decode JPEG to frame
            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
                
        except Exception as e:
            print(f"Error reading frame: {e}")
            
        return False, None
        
    def release(self):
        """Stop the capture process"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            
    def isOpened(self):
        return self.running and self.process is not None


class RpicamTcpCapture:
    """
    Capture frames using rpicam-vid with TCP output
    More reliable than pipe method
    """
    def __init__(self, width=640, height=480, fps=30, port=8888):
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.process = None
        self.cap = None
        self.running = False
        
    def start(self):
        """Start rpicam-vid with TCP output and connect OpenCV"""
        import socket
        
        # Find available port
        for port in range(self.port, self.port + 10):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:  # Port is available
                self.port = port
                break
        
        # Try rpicam-vid first, then libcamera-vid
        for cmd_name in ['rpicam-vid', 'libcamera-vid']:
            try:
                cmd = [
                    cmd_name, '--inline', '--nopreview', '-t', '0',
                    '--width', str(self.width), '--height', str(self.height),
                    '--framerate', str(self.fps),
                    '--codec', 'mjpeg',
                    '--listen', '-o', f'tcp://0.0.0.0:{self.port}'
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Wait for server to start
                time.sleep(2)
                
                # Connect OpenCV to TCP stream
                self.cap = cv2.VideoCapture(f'tcp://127.0.0.1:{self.port}')
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.running = True
                        print(f"{cmd_name} TCP started on port {self.port}: {self.width}x{self.height}@{self.fps}fps")
                        return True
                        
                # If failed, cleanup and try next
                if self.cap:
                    self.cap.release()
                if self.process:
                    self.process.terminate()
                    self.process.wait()
                    
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"TCP capture failed with {cmd_name}: {e}")
                continue
                
        return False
        
    def read(self):
        if self.cap and self.running:
            return self.cap.read()
        return False, None
        
    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()
            self.process.wait()
            
    def isOpened(self):
        return self.running and self.cap is not None and self.cap.isOpened()


class GStreamerCapture:
    """
    Capture frames using GStreamer pipeline with libcamera
    More efficient than raw pipe method
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def start(self):
        """Start GStreamer capture"""
        # GStreamer pipeline for libcamera on Pi5
        gst_pipeline = (
            f'libcamerasrc ! '
            f'video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! '
            f'videoconvert ! '
            f'video/x-raw,format=BGR ! '
            f'appsink drop=1'
        )
        
        try:
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                print(f"GStreamer capture started: {self.width}x{self.height}@{self.fps}fps")
                return True
        except Exception as e:
            print(f"GStreamer failed: {e}")
            
        return False
        
    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None
        
    def release(self):
        if self.cap:
            self.cap.release()
            
    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()


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
        self.camera_type = None  # 'picamera2', 'gstreamer', 'libcamera', 'opencv'
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
        print("\n" + "="*50)
        print("INITIALIZING CAMERA")
        print("="*50)
        
        # Method 1: Try picamera2
        if PICAMERA_AVAILABLE:
            try:
                print("Trying picamera2...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera_type = 'picamera2'
                print(f"✓ picamera2 initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                return
            except Exception as e:
                print(f"✗ picamera2 failed: {e}")
                self.camera = None
        
        # Method 2: Try GStreamer with libcamera
        print("Trying GStreamer with libcamera...")
        gst_cap = GStreamerCapture(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        if gst_cap.start():
            ret, test_frame = gst_cap.read()
            if ret and test_frame is not None:
                self.camera = gst_cap
                self.camera_type = 'gstreamer'
                print(f"✓ GStreamer initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                return
            gst_cap.release()
        print("✗ GStreamer failed")
        
        # Method 3: Try rpicam-vid with TCP (most reliable for Pi5)
        print("Trying rpicam-vid with TCP...")
        tcp_cap = RpicamTcpCapture(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        if tcp_cap.start():
            self.camera = tcp_cap
            self.camera_type = 'rpicam_tcp'
            print(f"✓ rpicam-vid TCP initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            return
        print("✗ rpicam-vid TCP failed")
        
        # Method 4: Try OpenCV with V4L2 (for Pi Camera)
        print("Trying OpenCV with V4L2...")
        for dev in ['/dev/video0', '/dev/video1', '/dev/video2']:
            try:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.camera = cap
                        self.camera_type = 'opencv'
                        print(f"✓ OpenCV V4L2 initialized on {dev}: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                        return
                    cap.release()
            except Exception as e:
                print(f"  {dev}: {e}")
                
        # Method 5: Try rpicam-vid/libcamera-vid pipe
        print("Trying rpicam-vid pipe...")
        libcam = LibcameraCapture(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        if libcam.start():
            time.sleep(1)  # Wait for camera to start
            ret, test_frame = libcam.read()
            if ret and test_frame is not None:
                self.camera = libcam
                self.camera_type = 'libcamera'
                print(f"✓ rpicam-vid pipe initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                return
            libcam.release()
        print("✗ rpicam-vid pipe failed")
                
        # Method 6: Try regular OpenCV
        print("Trying OpenCV default...")
        for camera_index in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.camera = cap
                        self.camera_type = 'opencv'
                        print(f"✓ OpenCV initialized on index {camera_index}")
                        return
                    cap.release()
            except:
                pass
                    
        print("\n" + "="*50)
        print("ERROR: Could not initialize any camera!")
        print("="*50)
        print("Please try:")
        print("  1. sudo apt install -y python3-picamera2 python3-libcamera")
        print("  2. pip install picamera2 (in venv with --system-site-packages)")
        print("  3. Check: ls /dev/video*")
        print("  4. Test: rpicam-hello -t 2000")
        print("="*50)
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
            if self.camera_type == 'picamera2':
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            else:
                # Works for opencv, gstreamer, libcamera
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
            if self.camera_type == 'picamera2':
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
        
        # Start camera if picamera2
        if self.camera_type == 'picamera2':
            self.camera.start()
            time.sleep(0.5)  # Wait for camera to warm up
        elif self.camera_type == 'libcamera':
            time.sleep(0.5)  # Already started, just wait
            
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
