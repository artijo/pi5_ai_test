# 🚌 Bus Passenger Counter

ระบบนับผู้โดยสารขึ้น-ลงรถบัส โดยใช้ Raspberry Pi 5 + AI Kit (Hailo-8L) + กล้อง

## 📋 คุณสมบัติ

- ✅ ตรวจจับคนแบบ Real-time ด้วย AI Kit (Hailo-8L)
- ✅ ติดตามคนด้วย Centroid Tracking
- ✅ นับคนเข้า-ออก แยกทิศทาง
- ✅ แสดงผลแบบ Real-time พร้อม visualization
- ✅ บันทึกผลลัพธ์เป็นไฟล์ `result.json`

## 🛠 อุปกรณ์ที่ต้องการ

- Raspberry Pi 5 (4GB/8GB)
- AI Kit (Hailo-8L)
- กล้อง (Pi Camera Module หรือ USB Camera)
- MicroSD Card (32GB+)
- Power Supply

## 📷 การติดตั้งกล้อง

กล้องต้องติดตั้งบนเพดานรถ มองลงมา:

```
      [กล้อง]
         |
         v
    ____________
   |            |
   |  ทางเข้า   |  <- คนเดินจากบน → ล่าง = เข้ารถ (IN)
   |   ประตู   |  <- คนเดินจากล่าง → บน = ออกรถ (OUT)
   |____________|
```

## 🚀 การติดตั้ง

### 1. ติดตั้ง Raspberry Pi OS

ใช้ Raspberry Pi Imager ติดตั้ง Raspberry Pi OS (64-bit) Bookworm

### 2. ติดตั้ง AI Kit (Hailo)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Hailo runtime
sudo apt install hailo-all -y

# Verify installation
hailortcli fw-control identify
```

### 3. ติดตั้ง Dependencies

```bash
# Clone project
git clone <your-repo-url>
cd pi5_ai_test

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install picamera2 (if using Pi Camera)
sudo apt install python3-picamera2 -y
```

### 4. ทดสอบกล้อง

```bash
# ทดสอบ rpicam-hello
rpicam-hello -t 5000

# หรือทดสอบด้วย libcamera
libcamera-hello -t 5000
```

## 📁 โครงสร้างโปรเจค

```
pi5_ai_test/
├── main.py              # โปรแกรมหลัก
├── requirements.txt     # Dependencies
├── README.md           # เอกสารนี้
└── src/
    ├── __init__.py
    ├── config.py       # ค่าคอนฟิก
    ├── tracker.py      # Centroid Tracker & Counter
    └── utils.py        # ฟังก์ชันช่วยเหลือ
```

## ⚙️ การตั้งค่า

แก้ไขไฟล์ `src/config.py`:

```python
# ความละเอียดกล้อง
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ค่า threshold สำหรับ detection
CONFIDENCE_THRESHOLD = 0.5

# ตำแหน่งเส้นนับ (0-1 = เปอร์เซ็นต์ของความสูงภาพ)
COUNTING_LINE_POSITION = 0.5  # กลางจอ

# โมเดล AI (เลือกตามที่มี)
HAILO_MODEL_PATH = "/usr/share/hailo-models/yolov5s_personface.hef"
```

## 🎮 การใช้งาน

### รันโปรแกรม

```bash
# เปิด virtual environment
source venv/bin/activate

# รันโปรแกรม
python main.py
```

### คีย์ลัด

| คีย์ | การทำงาน               |
| ---- | ---------------------- |
| `q`  | หยุดโปรแกรมและบันทึกผล |
| `r`  | รีเซ็ตตัวนับ           |

### ผลลัพธ์

เมื่อหยุดโปรแกรม จะสร้างไฟล์ `result.json`:

```json
{
  "session": {
    "start_time": "2025-12-09T10:00:00",
    "end_time": "2025-12-09T12:00:00",
    "duration_seconds": 7200
  },
  "counts": {
    "total_in": 45,
    "total_out": 32,
    "current_in_bus": 13
  },
  "summary": {
    "net_change": 13,
    "total_movements": 77
  },
  "events": [
    {
      "type": "in",
      "object_id": 1,
      "position": [320, 240],
      "timestamp": "2025-12-09T10:05:23"
    }
  ]
}
```

## 🔧 Troubleshooting

### ไม่พบ Hailo device

```bash
# ตรวจสอบ Hailo
lspci | grep Hailo
hailortcli fw-control identify
```

### กล้องไม่ทำงาน

```bash
# ตรวจสอบกล้อง
vcgencmd get_camera
libcamera-hello --list-cameras
```

### FPS ต่ำ

- ลดความละเอียดกล้องใน config
- ใช้โมเดลที่เล็กกว่า (yolov5n แทน yolov5s)
- ปิด visualization (`SHOW_PREVIEW = False`)

## 📊 โมเดล AI ที่รองรับ

โมเดลที่มาพร้อม Hailo:

| โมเดล   | ขนาด | ความเร็ว | ความแม่นยำ |
| ------- | ---- | -------- | ---------- |
| yolov5n | เล็ก | เร็วมาก  | ปานกลาง    |
| yolov5s | กลาง | เร็ว     | ดี         |
| yolov5m | ใหญ่ | ปานกลาง  | ดีมาก      |
| yolov8s | กลาง | เร็ว     | ดี         |

## 📝 License

MIT License

## 🙏 Credits

- Hailo AI - https://hailo.ai/
- Raspberry Pi Foundation
- OpenCV
