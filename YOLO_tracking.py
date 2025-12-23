import cv2
import sys
import time
import serial
import logging
import threading
import numpy as np
import tkinter as tk
import mysql.connector
from tkinter import ttk
import ttkbootstrap as ttkb
from ultralytics import YOLO
from datetime import datetime
from collections import deque
from PIL import Image, ImageTk
from tkinter import filedialog
from matplotlib.figure import Figure
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # Vision & YOLO
    YOLO_MODEL_PATH = "carton_model/weights/best.pt"
    CAMERA_INDEX = 1
    CONF_THRESHOLD = 0.85
    YOLO_CLASS_MAP = {0: "invalid", 1: "valid"}
    ERROR_CLASS_INDEX = 0
    ERROR_CLASS_NAME = YOLO_CLASS_MAP.get(ERROR_CLASS_INDEX, "Error")
    # ROI_COORDS = [200, 40, 400, 440]

    # Database
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASS = "Kien02022004@"
    DB_NAME = "conveyor_db"
    DB_PORT = 3306

    # Actuator/Serial
    SERIAL_PORT = 'COM10'
    BAUD_RATE = 4800
    SERVO_SIGNAL = 'E'
    CONVEYOR_START_SIGNAL = 'p1'
    CONVEYOR_STOP_SIGNAL = 'p0'

    # Tracking
    MAX_DISAPPEARED = 15
    MAX_DISTANCE = 80

class ErrorInfo:
    def __init__(self, track_id, product_id, error_type, center_x, center_y, bbox, timestamp):
        self.track_id = track_id
        self.product_id = product_id
        self.error_type = error_type
        self.center_x = center_x
        self.center_y = center_y
        self.bbox = bbox
        self.timestamp = timestamp

class ActuatorController(threading.Thread):
    def __init__(self, serial_port, baud_rate):
        super().__init__(daemon=True)
        self.running = True
        self.error_queue = deque()  # Hàng đợi các đối tượng ErrorInfo
        self.serial_conn = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # State tracking for servo and pump
        self.current_servo_angle = 90  # Default position
        self.pump_state = False
        
        self._setup_serial()

    def _setup_serial(self):
        """Thiết lập kết nối serial với ATmega16"""
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            time.sleep(2)  # Wait for connection to stabilize
            print(f"[Actuator] Serial connection established on {self.serial_port} at {self.baud_rate} baud.")
            
            # Read initial messages from ATmega16
            while self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                print(f"[Actuator] ATmega16: {response}")
            
            # Get initial status
            self.get_status()
            
        except Exception as e:
            print(f"[Actuator] WARNING: Cannot open serial port {self.serial_port}. Actuator is DISABLED. Error: {e}")
            self.serial_conn = None

    def _send_command(self, command):
        """
        Gửi lệnh đến ATmega16 và nhận phản hồi
        
        Args:
            command: Chuỗi lệnh (không cần '\n', sẽ tự động thêm)
        
        Returns:
            str: Phản hồi từ ATmega16 hoặc None nếu lỗi
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("[Actuator] Serial connection is not available.")
            return None
        
        try:
            # Send command
            self.serial_conn.write((command + '\n').encode('ascii'))
            time.sleep(0.1)  # Small delay for processing
            
            # Read response
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                print(f"[Actuator] Sent: '{command}' | Response: '{response}'")
                return response
            else:
                print(f"[Actuator] Sent: '{command}' | No response")
                return None
                
        except Exception as e:
            print(f"[Actuator] ERROR sending command '{command}': {e}")
            return None

    def _send_serial_signal(self, signal, description):
        """
        Phương thức tương thích với code cũ để gửi tín hiệu
        
        Args:
            signal: Tín hiệu cần gửi
            description: Mô tả tín hiệu
        
        Returns:
            bool: True nếu thành công
        """
        print(f"[Actuator] {description}: Sending '{signal}'")
        response = self._send_command(signal)
        return response is not None

    # ========== SERVO CONTROL ==========
    
    def set_servo(self, angle):
        """
        Điều khiển servo đến góc chỉ định
        
        Args:
            angle: Góc quay (0-180 độ)
        
        Returns:
            bool: True nếu thành công
        """
        if not 0 <= angle <= 180:
            print(f"[Actuator] ERROR: Servo angle must be between 0 and 180. Got: {angle}")
            return False
        
        command = f"S{angle}"
        response = self._send_command(command)
        
        if response and "OK" in response:
            self.current_servo_angle = angle
            return True
        return False
    
    def servo_home(self):
        """Di chuyển servo về vị trí home (90 độ)"""
        return self.set_servo(0)
    
    def servo_reject_position(self):
        """Di chuyển servo đến vị trí loại bỏ (có thể tùy chỉnh)"""
        # Tùy chỉnh góc này theo hệ thống của bạn
        return self.set_servo(120)  # Ví dụ: 45 độ là vị trí đẩy

    # ========== PUMP CONTROL ==========
    
    def set_pump(self, state):
        """
        Bật/tắt bơm
        
        Args:
            state: True để BẬT, False để TẮT
        
        Returns:
            bool: True nếu thành công
        """
        command = f"P{1 if state else 0}"
        response = self._send_command(command)
        
        if response and "OK" in response:
            self.pump_state = state
            return True
        return False
    
    def pump_on(self):
        """Bật bơm"""
        return self.set_pump(True)
    
    def pump_off(self):
        """Tắt bơm"""
        return self.set_pump(False)

    # ========== STATUS QUERY ==========
    
    def get_status(self):
        """
        Lấy trạng thái hiện tại của hệ thống từ ATmega16
        
        Returns:
            str: Chuỗi trạng thái hoặc None
        """
        response = self._send_command("?")
        return response

    # ========== CONVEYOR CONTROL (Giữ nguyên từ code cũ) ==========
    
    def start_conveyor(self):
        """Gửi tín hiệu chạy băng truyền."""
        # Sử dụng Config.CONVEYOR_START_SIGNAL từ code gốc
        # Hoặc có thể mapping sang lệnh ATmega16 nếu cần
        # from config import Config  # Import nếu cần
        return self._send_serial_signal(Config.CONVEYOR_START_SIGNAL, "START CONVEYOR")

    def stop_conveyor(self):
        """Gửi tín hiệu dừng băng truyền."""
        # from config import Config
        return self._send_serial_signal(Config.CONVEYOR_STOP_SIGNAL, "STOP CONVEYOR")

    # ========== ERROR HANDLING (Giữ nguyên logic cũ) ==========
    
    def add_error(self, error_info):
        """Thêm vật thể bị lỗi vào hàng đợi để chờ loại bỏ"""
        self.error_queue.append(error_info)
        print(f"[Actuator] Item {error_info.track_id} (Type: {error_info.error_type}) added to queue. Queue size: {len(self.error_queue)}")

    def run(self):
        """Vòng lặp chạy trong luồng Actuator"""
        while self.running:
            if self.error_queue:
                error_item = self.error_queue[0]  # Nhìn vào vật thể đầu tiên

                # Logic GIẢ ĐỊNH VỊ TRÍ LOẠI BỎ
                time_since_detection = (datetime.now() - error_item.timestamp).total_seconds()
                
                # THAM SỐ GIẢ ĐỊNH: Vật thể cần 5.0 giây để di chuyển từ điểm phát hiện đến Actuator
                TIME_TO_REACH_ACTUATOR = 3.5 

                if time_since_detection >= TIME_TO_REACH_ACTUATOR:
                    # Đã đến vị trí loại bỏ!
                    self.process_rejection(error_item)
                    self.error_queue.popleft()  # Loại bỏ khỏi hàng đợi sau khi xử lý
                else:
                    # Vẫn đang di chuyển, chờ đợi
                    time.sleep(0.5)  # Kiểm tra lại sau 0.5 giây
            else:
                time.sleep(0.1)  # Hàng đợi trống, chờ đợi

    def process_rejection(self, item):
        """
        Xử lý loại bỏ vật thể lỗi bằng servo và pump
        
        Có thể tùy chỉnh logic tùy theo error_type
        """
        print(f"[Actuator] --- REJECTING Track ID {item.track_id} at {datetime.now()} ---")
        
        if not self.serial_conn or not self.serial_conn.is_open:
            print("[Actuator] Serial connection is closed. Rejection FAILED.")
            return
        
        try:
            # PHƯƠNG ÁN 1: Sử dụng servo để đẩy vật thể ra
            # Di chuyển servo đến vị trí loại bỏ
            self.servo_reject_position()
            time.sleep(0.5)  # Chờ servo di chuyển
            
            # Quay về vị trí home
            self.servo_home()
            
            # PHƯƠNG ÁN 2: Sử dụng pump để hút/thổi vật thể
            # self.pump_on()
            # time.sleep(1.0)  # Bật pump trong 1 giây
            # self.pump_off()
            
            # PHƯƠNG ÁN 3: Sử dụng Config.SERVO_SIGNAL từ code cũ
            # from config import Config
            # self.serial_conn.write(Config.SERVO_SIGNAL.encode('ascii'))
            
            print(f"[Actuator] SUCCESS: Item {item.track_id} rejected.")
            
        except Exception as e:
            print(f"[Actuator] ERROR during rejection: {e}")

    def stop(self):
        """Dừng luồng và đóng kết nối"""
        print("[Actuator] Stopping actuator controller...")
        self.running = False
        
        # Reset thiết bị về trạng thái an toàn
        self.servo_home()
        self.pump_off()
        self.stop_conveyor()
        
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[Actuator] Serial connection closed.")

class SimpleCentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 1
        self.objects = {}
        self.bboxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.newly_registered = []

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.next_object_id += 1
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.newly_registered.append(oid)
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.bboxes[oid]
        del self.disappeared[oid]

    def update(self, rects):
        self.newly_registered = []
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {}

        input_centroids = []
        for (sX, sY, eX, eY) in rects:
            cX = int((sX + eX) / 2.0)
            cY = int((sY + eY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            assigned = {}
            for i, c in enumerate(input_centroids):
                oid = self.register(c, rects[i])
                assigned[oid] = rects[i]
            return assigned

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = np.linalg.norm(np.array(object_centroids)[:, None] - np.array(input_centroids)[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        assigned = {}

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.bboxes[oid] = rects[col]
            self.disappeared[oid] = 0
            assigned[oid] = rects[col]
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        unused_cols = set(range(0, D.shape[1])) - used_cols
        for col in unused_cols:
            oid = self.register(input_centroids[col], rects[col])
            assigned[oid] = rects[col]

        return assigned

class YOLOProcessor(threading.Thread):
    """
    Quản lý camera, chạy model YOLO, theo dõi vật thể và ra quyết định lỗi.
    """
    def __init__(self, actuator_controller: ActuatorController, initial_total=0, initial_invalid=0):
        super().__init__(daemon=True)
        self.running = False
        self.actuator = actuator_controller

        # Trạng thái
        self.current_frame = None
        # self.product_counter = 0
        # self.total_products_seen = 0 
        self.product_counter = initial_total        # để in STT nếu cần
        self.total_products_seen = initial_total    # tổng sản phẩm trong ngày
        self.invalid_count = initial_invalid        # số sản phẩm lỗi trong ngày
        self.logged_errors = set()
        self.logged_tracks = set()

        # Model/camera sẽ mở khi start_processing()
        self.model = None
        self.cap = None

        # Tracking
        self.tracker = SimpleCentroidTracker(max_disappeared=Config.MAX_DISAPPEARED,
                                             max_distance=Config.MAX_DISTANCE)
        self.track_classes = {}
        # self.ROI_X_MIN, self.ROI_Y_MIN, self.ROI_X_MAX, self.ROI_Y_MAX = Config.ROI_COORDS

        # Tốc độ detect
        self.detection_interval = 0.05
        self.last_detection_time = 0
        self.detect_lock = threading.Lock()

    def _setup_vision(self):
        """Tải model YOLO và mở camera (chỉ gọi khi start)."""
        if self.model is None:
            try:
                self.model = YOLO(Config.YOLO_MODEL_PATH)
                print(f"[YOLOProcessor] Loaded YOLO model: {Config.YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"[YOLOProcessor] FATAL: Cannot load YOLO model: {e}")
                raise

        if self.cap is None:
            cap = cv2.VideoCapture(Config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(Config.CAMERA_INDEX, cv2.CAP_MSMF)
            if not cap.isOpened():
                raise IOError("Cannot open camera.")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap

    def start_processing(self):
        """Khởi động luồng xử lý."""
        self._setup_vision()
        self.running = True
        self.start()

    def stop_processing(self):
        """Dừng luồng xử lý; camera giải phóng trong finally của run()."""
        self.running = False

    def run(self):
        try:
            while self.running:
                ret, frame = self.cap.read() if self.cap is not None else (False, None)
                if not ret:
                    time.sleep(0.05)
                    continue
                self.current_frame = frame.copy()
                self._process_frame(frame)
                time.sleep(0.01)
        finally:
            if getattr(self, "cap", None) is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _process_frame(self, frame):
        now = time.time()
        if now - self.last_detection_time < self.detection_interval:
            return

        if not self.detect_lock.acquire(blocking=False):
            return

        try:
            self.last_detection_time = now
            conf = Config.CONF_THRESHOLD
            results = self.model(frame, conf=conf, verbose=False)
            boxes, classes = self._extract_detections(results)

            assigned_tracks = self.tracker.update(boxes)
            bbox_to_class = dict(zip(boxes, classes))
            # self.total_products_seen += len(self.tracker.newly_registered)

            for tid, bbox in assigned_tracks.items():
                self._handle_track_update(tid, bbox, bbox_to_class)
        except Exception as e:
            print(f"[YOLOProcessor] Detection error: {e}")
        finally:
            self.detect_lock.release()

    def _extract_detections(self, results):
        boxes, classes = [], []
        r = results[0] if isinstance(results, list) else results

        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            return boxes, classes

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            if cls in Config.YOLO_CLASS_MAP:
                boxes.append((x1, y1, x2, y2))
                classes.append(cls)
        return boxes, classes

    def _handle_track_update(self, tid, bbox, bbox_to_class):
        sX, sY, eX, eY = bbox
        cx = int((sX + eX) / 2)
        cy = int((sY + eY) / 2)

        detected_cls = bbox_to_class.get(bbox, 0)
        self.track_classes[tid] = detected_cls

        # Loại sản phẩm (valid / invalid)
        product_type = Config.YOLO_CLASS_MAP.get(detected_cls, "Unknown")

        # Mỗi track chỉ log 1 lần (trong lần chạy hiện tại)
        if tid in self.logged_tracks:
            return

        # Tăng STT & bộ đếm tổng
        self.product_counter += 1
        self.total_products_seen += 1

        error_data = ErrorInfo(
            track_id=tid,
            product_id=self.product_counter,
            error_type=product_type,      # "valid" hoặc "invalid"
            center_x=cx,
            center_y=cy,
            bbox=bbox,
            timestamp=datetime.now()
        )

        # Nếu là invalid → đưa vào hàng đợi actuator + đếm lỗi
        if detected_cls == Config.ERROR_CLASS_INDEX:
            print(f"[YOLOProcessor] NEW INVALID PRODUCT: Track ID {tid} (STT={self.product_counter})")
            self.actuator.add_error(error_data)
            self.logged_errors.add(tid)
            self.invalid_count += 1   # tăng số lỗi trong ngày
        else:
            print(f"[YOLOProcessor] NEW VALID PRODUCT: Track ID {tid} (STT={self.product_counter})")

        # Ghi vào database (cả valid & invalid)
        try:
            if hasattr(self, "log_error") and callable(self.log_error):
                self.log_error(error_data)
        except Exception as e:
            print(f"[YOLOProcessor] DB log error: {e}")

        # Đánh dấu track này đã được log trong lần chạy hiện tại
        self.logged_tracks.add(tid)

class DatabaseManager:
    def __init__(self):
        self._ensure_db_and_table()
        print("[DB Manager] Database and table structure confirmed.")

    def get_today_counters(self):
        """
        Trả về (total_products_today, invalid_products_today)
        dựa trên dữ liệu trong bảng errors trong NGÀY HÔM NAY.
        """
        try:
            conn = self._connect()
            cur = conn.cursor()
            invalid_name = Config.YOLO_CLASS_MAP.get(Config.ERROR_CLASS_INDEX, "invalid")

            cur.execute("""
                SELECT
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN error_type = %s THEN 1 ELSE 0 END) AS invalid_count
                FROM errors
                WHERE DATE(timestamp) = CURDATE()
            """, (invalid_name,))

            row = cur.fetchone()
            cur.close(); conn.close()

            if not row:
                return 0, 0

            total = row[0] if row[0] is not None else 0
            invalid = row[1] if row[1] is not None else 0
            return total, invalid

        except Exception as e:
            print(f"[DB Manager] get_today_counters error: {e}")
            return 0, 0

    def _ensure_db_and_table(self):
        try:
            conn = mysql.connector.connect(host=Config.DB_HOST, user=Config.DB_USER, password=Config.DB_PASS)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME}")
            conn.commit()
            conn.close()

            conn = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASS,
                database=Config.DB_NAME
            )
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    error_type VARCHAR(100),
                    timestamp DATETIME
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Manager] CRITICAL DB ERROR during setup: {e}")

    def log_error(self, error_info: ErrorInfo):
        try:
            conn = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASS,
                database=Config.DB_NAME
            )
            cursor = conn.cursor()
            sql = "INSERT INTO errors (error_type, timestamp) VALUES (%s, %s)"
            cursor.execute(sql, (
                error_info.error_type,   # "valid" hoặc "invalid"
                error_info.timestamp
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Manager] DB insert error: {e}")


    def _connect(self):
        return mysql.connector.connect(
            host=Config.DB_HOST, user=Config.DB_USER,
            password=Config.DB_PASS, database=Config.DB_NAME
        )

    def fetch_errors_range(self, start_dt=None, end_dt=None):
        """
        Lấy tất cả bản ghi trong khoảng [start_dt, end_dt].
        Nếu không truyền, sẽ trả toàn bộ.
        start_dt/end_dt: datetime hoặc string 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM:SS'
        """
        try:
            conn = self._connect()
            cur = conn.cursor(dictionary=True)
            if start_dt and end_dt:
                cur.execute(
                    "SELECT * FROM errors WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp DESC",
                    (start_dt, end_dt)
                )
            elif start_dt:
                cur.execute(
                    "SELECT * FROM errors WHERE timestamp >= %s ORDER BY timestamp DESC",
                    (start_dt,)
                )
            elif end_dt:
                cur.execute(
                    "SELECT * FROM errors WHERE timestamp <= %s ORDER BY timestamp DESC",
                    (end_dt,)
                )
            else:
                cur.execute("SELECT * FROM errors ORDER BY timestamp DESC")
            rows = cur.fetchall()
            cur.close(); conn.close()
            return rows
        except Exception as e:
            print(f"[DB Manager] fetch_errors_range error: {e}")
            return []

    def fetch_errors_by_day(self, day_str):
        """day_str: 'YYYY-MM-DD'"""
        try:
            conn = self._connect()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT * FROM errors WHERE DATE(timestamp) = %s ORDER BY timestamp DESC",
                (day_str,)
            )
            rows = cur.fetchall()
            cur.close(); conn.close()
            return rows
        except Exception as e:
            print(f"[DB Manager] fetch_errors_by_day error: {e}")
            return []

    def fetch_errors_by_month(self, year: int, month: int):
        try:
            conn = self._connect()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT * FROM errors WHERE YEAR(timestamp)=%s AND MONTH(timestamp)=%s ORDER BY timestamp DESC",
                (year, month)
            )
            rows = cur.fetchall()
            cur.close(); conn.close()
            return rows
        except Exception as e:
            print(f"[DB Manager] fetch_errors_by_month error: {e}")
            return []

    def fetch_errors_by_year(self, year: int):
        try:
            conn = self._connect()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT * FROM errors WHERE YEAR(timestamp)=%s ORDER BY timestamp DESC",
                (year,)
            )
            rows = cur.fetchall()
            cur.close(); conn.close()
            return rows
        except Exception as e:
            print(f"[DB Manager] fetch_errors_by_year error: {e}")
            return []

class ConveyorApp(ttkb.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Conveyor Monitoring System")
        self.geometry("1000x700")


        self.db_manager = DatabaseManager()
        self.actuator_controller = ActuatorController(Config.SERIAL_PORT, Config.BAUD_RATE)

        today_total, today_invalid = self.db_manager.get_today_counters()

        # Processor sẽ được tạo khi nhấn Start
        self.processor = None

        self.total_products = tk.IntVar(value=today_total)
        self.error_count = tk.IntVar(value=today_invalid)

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Bắt đầu vòng lặp cập nhật UI
        self._update_ui_loop()

    def _create_widgets(self):
        # Notebook (tabs)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=BOTH, expand=YES)

        # Tab 1: Monitor (code cũ chuyển vào hàm riêng)
        self.monitor_tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.monitor_tab, text="Monitor")
        self._create_monitor_tab(self.monitor_tab)

        # Tab 2: Database
        self.db_tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.db_tab, text="Database")
        self._create_db_tab(self.db_tab)

        # Tab 3: Statistics
        self.stats_tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.stats_tab, text="Statistics")
        self._create_stats_tab(self.stats_tab)

    def _create_monitor_tab(self, parent):
        # Layout chính (2 cột)
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=BOTH, expand=YES)

        # === VIDEO (Cột trái) ===
        video_frame = ttk.LabelFrame(main_frame, text="Live Monitoring", padding=10)
        video_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=BOTH, expand=YES)

        self.placeholder = ImageTk.PhotoImage(Image.new("RGB", (640, 480), "gray"))
        self.video_label.configure(image=self.placeholder)

        # === Thống kê + Log (Cột phải) ===
        stats_log_frame = ttk.Frame(main_frame)
        stats_log_frame.pack(side=RIGHT, fill=Y, padx=(10, 0))

        # Stats
        stats_frame = ttk.LabelFrame(stats_log_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=X, pady=(0, 10))

        ttk.Label(stats_frame, text="Total Products:", font='Helvetica 10 bold').grid(row=0, column=0, sticky=W, padx=5, pady=5)
        ttk.Label(stats_frame, textvariable=self.total_products, font='Helvetica 10', bootstyle="primary").grid(row=0, column=1, sticky=W, padx=5, pady=5)

        ttk.Label(stats_frame, text="Error Count:", font='Helvetica 10 bold').grid(row=1, column=0, sticky=W, padx=5, pady=5)
        ttk.Label(stats_frame, textvariable=self.error_count, font='Helvetica 10', bootstyle="danger").grid(row=1, column=1, sticky=W, padx=5, pady=5)

        # Log
        log_frame = ttk.LabelFrame(stats_log_frame, text="Real-time Log", padding=10)
        log_frame.pack(fill=BOTH, expand=YES, pady=(0, 10))

        self.log_listbox = tk.Listbox(log_frame, height=20, width=40, font='TkFixedFont', bg="#333", fg="#fff", highlightthickness=0)
        self.log_listbox.pack(fill=BOTH, expand=YES)

        # Buttons
        control_frame = ttk.Frame(stats_log_frame, padding=5)
        control_frame.pack(fill=X)

        self.btn_start = ttk.Button(control_frame, text="Start System", command=self._start_system, bootstyle=SUCCESS)
        self.btn_start.pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))

        self.btn_stop = ttk.Button(control_frame, text="Stop System", command=self._stop_system, bootstyle=DANGER, state=DISABLED)
        self.btn_stop.pack(side=LEFT, fill=X, expand=YES, padx=(5, 0))

    def _create_db_tab(self, parent):
        # Top controls
        control = ttk.LabelFrame(parent, text="Filters", padding=10)
        control.pack(fill=X)

        # Mode chọn: Day/Month/Year/Range
        self.db_mode = tk.StringVar(value="day")
        modes = [("Day", "day"), ("Month", "month"), ("Year", "year"), ("Range", "range")]
        for i, (txt, val) in enumerate(modes):
            ttk.Radiobutton(control, text=txt, value=val, variable=self.db_mode).grid(row=0, column=i, padx=6, pady=6, sticky=W)

        # Day picker
        ttk.Label(control, text="Day:").grid(row=1, column=0, sticky=E, padx=6, pady=2)
        self.day_picker = DateEntry(control, bootstyle="primary", dateformat="%Y-%m-%d")
        self.day_picker.grid(row=1, column=1, sticky=W, padx=6, pady=2)

        # Month/Year picker
        ttk.Label(control, text="Month:").grid(row=1, column=2, sticky=E, padx=6, pady=2)
        self.month_spin = ttk.Spinbox(control, from_=1, to=12, width=5)
        self.month_spin.grid(row=1, column=3, sticky=W, padx=6, pady=2)
        self.month_spin.insert(0, datetime.now().month)

        ttk.Label(control, text="Year:").grid(row=1, column=4, sticky=E, padx=6, pady=2)
        self.year_spin = ttk.Spinbox(control, from_=2000, to=2100, width=6)
        self.year_spin.grid(row=1, column=5, sticky=W, padx=6, pady=2)
        self.year_spin.insert(0, datetime.now().year)

        # Range pickers
        ttk.Label(control, text="From:").grid(row=2, column=0, sticky=E, padx=6, pady=2)
        self.range_from = DateEntry(control, bootstyle="secondary", dateformat="%Y-%m-%d")
        self.range_from.grid(row=2, column=1, sticky=W, padx=6, pady=2)

        ttk.Label(control, text="To:").grid(row=2, column=2, sticky=E, padx=6, pady=2)
        self.range_to = DateEntry(control, bootstyle="secondary", dateformat="%Y-%m-%d")
        self.range_to.grid(row=2, column=3, sticky=W, padx=6, pady=2)

        # Buttons
        ttk.Button(control, text="Search", bootstyle=PRIMARY, command=self._db_search).grid(row=0, column=4, padx=6, pady=6, sticky=E)
        ttk.Button(control, text="Reset", bootstyle=INFO, command=self._db_reset).grid(row=0, column=5, padx=6, pady=6, sticky=W)
        ttk.Button(control, text="Export CSV", bootstyle=SUCCESS, command=self._db_export_csv).grid(row=0, column=6, padx=6, pady=6, sticky=W)

        # Table
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=BOTH, expand=YES, pady=(10,0))

        cols = ("id", "error_type", "timestamp")
        self.db_tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            height=18,
            bootstyle="dark"
        )

        self.db_tree.heading("id", text="STT")
        self.db_tree.heading("error_type", text="Loại sản phẩm")
        self.db_tree.heading("timestamp", text="Thời gian phát hiện")

        self.db_tree.column("id", anchor=CENTER, width=80)
        self.db_tree.column("error_type", anchor=CENTER, width=150)
        self.db_tree.column("timestamp", anchor=CENTER, width=180)

        self.db_tree.pack(fill=BOTH, expand=YES)

        # Load all on open
        self._db_load_all()

    def _create_stats_tab(self, parent):
        # Bộ lọc chọn loại thống kê
        top = ttk.LabelFrame(parent, text="Statistics Options", padding=10)
        top.pack(fill=X)

        self.stats_mode = tk.StringVar(value="day")
        ttk.Label(top, text="Group by:").pack(side=LEFT, padx=(5,10))
        ttk.Radiobutton(top, text="Day", variable=self.stats_mode, value="day").pack(side=LEFT, padx=5)
        ttk.Radiobutton(top, text="Month", variable=self.stats_mode, value="month").pack(side=LEFT, padx=5)
        ttk.Radiobutton(top, text="Year", variable=self.stats_mode, value="year").pack(side=LEFT, padx=5)

        ttk.Button(top, text="Generate Chart", bootstyle=PRIMARY, command=self._generate_stats_chart).pack(side=LEFT, padx=10)
        ttk.Button(top, text="Export Chart", bootstyle=SUCCESS, command=self._export_chart_image).pack(side=LEFT, padx=10)

        # Khung biểu đồ
        frame_chart = ttk.Frame(parent)
        frame_chart.pack(fill=BOTH, expand=YES, pady=(10,0))
        self.chart_frame = frame_chart
        self.stats_canvas = None

    def _generate_stats_chart(self):
        mode = self.stats_mode.get()
        conn = None
        try:
            conn = self.db_manager._connect()
            cur = conn.cursor()

            if mode == "day":
                query = """
                    SELECT DATE(timestamp) AS label, COUNT(*) AS count
                    FROM errors
                    GROUP BY DATE(timestamp)
                    ORDER BY DATE(timestamp)
                """
            elif mode == "month":
                query = """
                    SELECT DATE_FORMAT(timestamp, '%Y-%m') AS label, COUNT(*) AS count
                    FROM errors
                    GROUP BY YEAR(timestamp), MONTH(timestamp)
                    ORDER BY YEAR(timestamp), MONTH(timestamp)
                """
            else:  # year
                query = """
                    SELECT YEAR(timestamp) AS label, COUNT(*) AS count
                    FROM errors
                    GROUP BY YEAR(timestamp)
                    ORDER BY YEAR(timestamp)
                """

            cur.execute(query)
            data = cur.fetchall()
            cur.close(); conn.close()

            if not data:
                self._add_to_log("[Stats] No data to display.")
                return

            labels = [str(row[0]) for row in data]
            counts = [row[1] for row in data]

            # Vẽ biểu đồ
            fig = Figure(figsize=(8,4), dpi=100)
            ax = fig.add_subplot(111)
            ax.bar(labels, counts, color='tomato')
            # ax.set_title(f"Error count grouped by {mode}")
            ax.set_title(f"Record count grouped by {mode}")
            ax.set_xlabel(mode.capitalize())
            # ax.set_ylabel("Number of errors")
            ax.set_ylabel("Number of records")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Clear chart cũ
            for widget in self.chart_frame.winfo_children():
                widget.destroy()

            # Nhúng matplotlib vào Tkinter
            self.stats_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            self.stats_canvas.draw()
            self.stats_canvas.get_tk_widget().pack(fill=BOTH, expand=YES)
            self._add_to_log(f"[Stats] Chart generated for {mode}-level summary.")
        except Exception as e:
            if conn: conn.close()
            self._add_to_log(f"[Stats] Error generating chart: {e}")

    def _export_chart_image(self):
        if not self.stats_canvas:
            self._add_to_log("[Stats] No chart to export.")
            return
        fp = filedialog.asksaveasfilename(
            title="Export Chart Image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not fp:
            return
        try:
            self.stats_canvas.figure.savefig(fp, bbox_inches="tight")
            self._add_to_log(f"[Stats] Chart saved to {fp}")
        except Exception as e:
            self._add_to_log(f"[Stats] Export error: {e}")

    def _db_load_all(self):
        rows = self.db_manager.fetch_errors_range()
        self._db_fill_table(rows)

    def _db_fill_table(self, rows):
        # clear
        for i in self.db_tree.get_children():
            self.db_tree.delete(i)

        # fill: chỉ còn id - loại sản phẩm - thời gian phát hiện
        for r in rows:
            ts = r.get("timestamp")
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            self.db_tree.insert("", tk.END, values=(
                r.get("id"),
                r.get("error_type"),  # valid / invalid
                ts_str
            ))

    def _db_search(self):
        mode = self.db_mode.get()
        try:
            if mode == "day":
                day = self.day_picker.entry.get()  # 'YYYY-MM-DD'
                rows = self.db_manager.fetch_errors_by_day(day)
            elif mode == "month":
                m = int(self.month_spin.get())
                y = int(self.year_spin.get())
                rows = self.db_manager.fetch_errors_by_month(y, m)
            elif mode == "year":
                y = int(self.year_spin.get())
                rows = self.db_manager.fetch_errors_by_year(y)
            else:  # range
                start = self.range_from.entry.get()
                end = self.range_to.entry.get()
                # bao phủ hết ngày 'end' → thêm 23:59:59
                rows = self.db_manager.fetch_errors_range(
                    f"{start} 00:00:00", f"{end} 23:59:59"
                )
            self._db_fill_table(rows)
        except Exception as e:
            self._add_to_log(f"[DB] Search error: {e}")

    def _db_reset(self):
        self.db_mode.set("day")
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            self.day_picker.entry.delete(0, tk.END); self.day_picker.entry.insert(0, today)
            self.month_spin.delete(0, tk.END); self.month_spin.insert(0, datetime.now().month)
            self.year_spin.delete(0, tk.END); self.year_spin.insert(0, datetime.now().year)
            self.range_from.entry.delete(0, tk.END); self.range_from.entry.insert(0, today)
            self.range_to.entry.delete(0, tk.END); self.range_to.entry.insert(0, today)
        except Exception:
            pass
        self._db_load_all()

    def _db_export_csv(self):
        # Lấy dữ liệu hiện có trong Treeview
        rows = []
        for iid in self.db_tree.get_children():
            rows.append(self.db_tree.item(iid)["values"])

        if not rows:
            self._add_to_log("[DB] Nothing to export.")
            return

        fp = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv")]
        )
        if not fp:
            return

        try:
            import csv
            # cols = ("id","product_id","track_id","error_type","x_center","y_center","timestamp")
            # with open(fp, "w", newline="", encoding="utf-8") as f:
            #     writer = csv.writer(f)
            #     writer.writerow(cols)
            #     writer.writerows(rows)
            cols = ("STT", "Loai san pham", "Thoi gian phat hien")
            with open(fp, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)

            self._add_to_log(f"[DB] Exported CSV: {fp}")
        except Exception as e:
            self._add_to_log(f"[DB] Export error: {e}")

    def _update_ui_loop(self):
        self._update_video_frame()
        self._update_stats()
        self.after(30, self._update_ui_loop)

    def _update_video_frame(self):
        frame = None
        if getattr(self, "processor", None) is not None:
            frame = getattr(self.processor, "current_frame", None)

        if frame is not None:
            frame_to_display = self._draw_detections(frame.copy())
            cv2image = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            self.video_label.configure(image=self.placeholder)
            self.video_label.imgtk = self.placeholder

    def _update_stats(self):
        if self.processor is not None:
            # Tổng sản phẩm trong ngày (valid + invalid)
            if self.total_products.get() != self.processor.total_products_seen:
                self.total_products.set(self.processor.total_products_seen)

            # Số sản phẩm lỗi trong ngày
            current_errors = self.processor.invalid_count
            if self.error_count.get() != current_errors:
                self.error_count.set(current_errors)
                self._add_to_log(f"[ERROR LOGGED] Total errors today: {current_errors}")


    # def _draw_detections(self, frame):
    #     assigned_tracks = self.processor.tracker.bboxes.items() if self.processor else []
    #     track_classes = self.processor.track_classes if self.processor else {}

    #     for tid, bbox in assigned_tracks:
    #         sX, sY, eX, eY = bbox
    #         cls = track_classes.get(tid, 0)
    #         color = (0, 0, 255) if cls == Config.ERROR_CLASS_INDEX else (0, 255, 0)
    #         text = f"ID: {tid} ({Config.YOLO_CLASS_MAP.get(cls, 'Unknown')})"
    #         cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
    #         cv2.putText(frame, text, (sX, sY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #     return frame

    def _draw_detections(self, frame):
        # Lấy processor hiện tại
        proc = getattr(self, "processor", None)
        if proc is None:
            return frame

        try:
            # CHỤP SNAPSHOT: copy ra list/dict riêng để tránh bị thay đổi giữa chừng
            bboxes_items = list(proc.tracker.bboxes.items())
            track_classes = dict(proc.track_classes)
        except Exception:
            # Nếu có vấn đề trong lúc copy (ví dụ processor đang shutdown) thì bỏ qua
            return frame

        for tid, bbox in bboxes_items:
            sX, sY, eX, eY = bbox
            cls = track_classes.get(tid, 0)
            color = (0, 0, 255) if cls == Config.ERROR_CLASS_INDEX else (0, 255, 0)
            text = f"ID: {tid} ({Config.YOLO_CLASS_MAP.get(cls, 'Unknown')})"
            cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
            cv2.putText(
                frame,
                text,
                (sX, sY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return frame

    def _add_to_log(self, message):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.log_listbox.insert(tk.END, f"[{current_time}] {message}")
        self.log_listbox.see(tk.END)

    def _start_system(self):
        # Bật actuator thread nếu chưa chạy
        if not self.actuator_controller.is_alive():
            self.actuator_controller.start()

        if self.actuator_controller.start_conveyor():
            self._add_to_log(f"SENT SIGNAL: Start Conveyor ('{Config.CONVEYOR_START_SIGNAL}')")
        else:
            self._add_to_log("WARNING: Could not send Start Conveyor signal (serial unavailable).")

        if (getattr(self, "processor", None) is None) or (not self.processor.is_alive()):
            try:
                initial_total = self.total_products.get()
                initial_invalid = self.error_count.get()
                self.processor = YOLOProcessor(self.actuator_controller,initial_total=initial_total, initial_invalid=initial_invalid)
                self.processor.log_error = self.db_manager.log_error
                time.sleep(0.1)
                self.processor.start_processing()
                self._add_to_log("YOLO Processor started.")
                self.btn_start.configure(state=tk.DISABLED)
                self.btn_stop.configure(state=tk.NORMAL)
            except Exception as e:
                self._add_to_log(f"FATAL: cannot start YOLO/camera: {e}")
                self.btn_start.configure(state=tk.NORMAL)
                self.btn_stop.configure(state=tk.DISABLED)
        else:
            self._add_to_log("System already running.")

    def _stop_system(self):
        if self.actuator_controller:
            if self.actuator_controller.stop_conveyor():
                self._add_to_log(f"SENT SIGNAL: Stop Conveyor ('{Config.CONVEYOR_STOP_SIGNAL}')")
            else:
                self._add_to_log("WARNING: Could not send Stop Conveyor signal.")

        if getattr(self, "processor", None) and self.processor.is_alive():
            self.processor.stop_processing()
            self.processor.join(timeout=2.0)

        self.processor = None
        self.video_label.configure(image=self.placeholder)
        self.video_label.imgtk = self.placeholder
        time.sleep(0.3)

        self._add_to_log("YOLO Processor stopped.")
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)

    def _on_closing(self):
        try:
            if getattr(self, "processor", None) is not None:
                self.processor.stop_processing()
                if self.processor.is_alive():
                    self.processor.join(timeout=2)

            if getattr(self, "actuator_controller", None) is not None:
                if self.actuator_controller.is_alive():
                    self.actuator_controller.stop()
                    self.actuator_controller.join(timeout=2)
        finally:
            self.destroy()

def main():
    print("--- Starting Conveyor Logic System ---")
    try:
        db_manager = DatabaseManager()
    except Exception:
        print("FATAL: Cannot initialize Database Manager. Exiting.")
        sys.exit(1)

    actuator_controller = ActuatorController(Config.SERIAL_PORT, Config.BAUD_RATE)
    actuator_controller.start()
    time.sleep(1)

    yolo_processor = YOLOProcessor(actuator_controller=actuator_controller)
    yolo_processor.log_error = db_manager.log_error

    try:
        yolo_processor.start_processing()
        print("\n*** System is RUNNING. Press Ctrl+C to stop. ***\n")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred in main thread: {e}")
    finally:
        print("Stopping YOLO Processor...")
        yolo_processor.stop_processing()

        print("Stopping Actuator Controller...")
        actuator_controller.stop()

        yolo_processor.join(timeout=2)
        actuator_controller.join(timeout=2)

        print("Shutdown complete. Goodbye.")

if __name__ == "__main__":
    # main()
    app = ConveyorApp()
    app.mainloop()