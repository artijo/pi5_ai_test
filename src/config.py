"""
Configuration settings for Bus Passenger Counter
"""

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold

# Tracking settings
MAX_DISAPPEARED = 30  # Max frames before object is deregistered
MAX_DISTANCE = 50  # Max distance for centroid matching

# Counting line settings (horizontal line - percentage of frame height)
# Since camera is mounted on top, we use a horizontal counting line
COUNTING_LINE_POSITION = 0.5  # 50% of frame height (middle)
COUNTING_LINE_MARGIN = 20  # Pixels margin for counting zone

# Direction settings
# IN: moving from top to bottom (entering bus)
# OUT: moving from bottom to top (exiting bus)
DIRECTION_IN = "in"
DIRECTION_OUT = "out"

# AI Kit / Hailo settings
HAILO_MODEL_PATH = "/usr/share/hailo-models/yolov5s_personface.hef"
# Alternative models:
# HAILO_MODEL_PATH = "/usr/share/hailo-models/yolov8s.hef"
# HAILO_MODEL_PATH = "/usr/share/hailo-models/yolov5m_wo_spp.hef"

# Person class ID (COCO dataset)
PERSON_CLASS_ID = 0

# Output settings
OUTPUT_JSON_FILE = "result.json"

# Display settings
SHOW_PREVIEW = True
SHOW_TRACKING_INFO = True
DRAW_COUNTING_LINE = True
DRAW_BOUNDING_BOXES = True
DRAW_CENTROIDS = True
DRAW_TRAILS = True
TRAIL_LENGTH = 30  # Number of points in trail
