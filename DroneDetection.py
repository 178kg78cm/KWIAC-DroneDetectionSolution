import sys
import logging
import os

from contextlib import contextmanager
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, QSizePolicy, QTextEdit, QDesktopWidget, QInputDialog, QProgressBar
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from datetime import datetime, timedelta
import torch
logging.getLogger("sahi").setLevel(logging.ERROR)

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class VideoProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, file_path, frame_rate, detection_model, cap):
        super().__init__()
        self.file_path = file_path
        self.frame_rate = frame_rate
        self.detection_model = detection_model
        self.cap = cap
        self.out = None
        self.is_video = self.cap.isOpened()  
        self.save_path = None

    def run(self):
        save_dir = os.path.join(os.getcwd(), 'file_detect')
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(self.file_path))[0]

        if self.is_video:
            self._setup_video_writer(save_dir, file_name)
            self._process_video()
        else:
            self._process_image(save_dir, file_name)

        self.finished_signal.emit()

    def _setup_video_writer(self, save_dir, file_name):
        self.save_path = os.path.join(save_dir, f"{file_name}_detect.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.save_path, fourcc, fps, (frame_width, frame_height))

    def _process_video(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.frame_rate and frame_counter % self.frame_rate != 0:
                frame_counter += 1
                continue

            self._process_frame(frame)
            frame_counter += 1
            self.progress_signal.emit(int((frame_counter / total_frames) * 100))

        self.cap.release()
        if self.out:
            self.out.release()

    def _process_image(self, save_dir, file_name):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self._process_frame(frame)

        self.save_path = os.path.join(save_dir, f"{file_name}_detect.jpg")
        cv2.imwrite(self.save_path, frame)

    def _process_frame(self, frame):
        # 이미지가 4채널(RGBA)인 경우 3채널(BGR)로 변환
        if frame.shape[2] == 4:  # RGBA 형식인지 확인
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 1:  # 그레이스케일인 경우 BGR로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        with suppress_stdout():
            results = get_sliced_prediction(
                frame,
                self.detection_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )

        for object_prediction in results.object_prediction_list:
            bbox = object_prediction.bbox
            category_name = object_prediction.category.name
            cv2.rectangle(frame, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (0, 255, 0), 2)
            cv2.putText(frame, category_name, (int(bbox.minx), int(bbox.miny) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if self.is_video:
            self.out.write(frame)


    def terminate(self):
        """ 스레드 종료 시 파일을 안전하게 닫고 삭제 """
        if self.out:
            self.out.release()
        if self.save_path and os.path.exists(self.save_path):
            os.remove(self.save_path)
        super().terminate()

class DroneDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)

        # best.pt와 kw_logo.png의 경로 설정
        self.model_path = os.path.join(base_path, 'best.pt')
        self.icon_path = os.path.join(base_path, 'kw_logo.png')
        
        self.setWindowTitle('Drone Detection System')
        self.setGeometry(100, 100, 400, 300)
        self.setWindowIcon(QIcon(self.icon_path))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.log_file = 'timeline_log.txt'
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=self.model_path,
            confidence_threshold=0.3,
            device=device
        )
        self.drone_detected = False
        self.start_time = None
        self.last_detection_time = None
        self.timeline_content = ""
        self.camera_on = False
        self.progress_bar = None

        self.setStyleSheet("""
            QPushButton {
                background-color: #92cec6;  
                border: none;
                color: black;
                font-size: 18px;
                padding: 10px;
                margin: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #308486;  
            }
            QPushButton:pressed {
                background-color: #93cdc8;  
            }
            QLabel {
                font-size: 18px;
                color: #333;
                padding: 5px;
            }
        """)
        self.init_ui()

    def init_ui(self):
        self.clear_layout()
        self.resize(400, 300)

        main_layout = QVBoxLayout()

        self.label = QLabel('Choose Mode')
        self.label.setFont(QFont("Arial", 16))
        self.label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label)

        file_detect_button = QPushButton('1. File Detection', self)
        file_detect_button.clicked.connect(self.detect_from_file)
        main_layout.addWidget(file_detect_button)

        camera_detect_button = QPushButton('2. Camera Detection', self)
        camera_detect_button.clicked.connect(self.show_camera_options)
        main_layout.addWidget(camera_detect_button)

        self.setLayout(main_layout)
        self.center_window()

    def center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def detect_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a video or image file", "", "Video Files (*.mp4 *.avi);;Image Files (*.jpg *.png)")
        if not file_path:
            QMessageBox.information(self, "Error", "No file selected")
            return

        if file_path.endswith('.mp4') or file_path.endswith('.avi'):
            frame_rate, ok = QInputDialog.getInt(self, 'Frame Rate Input', 'Enter the frame rate to process', min=1, max=60)
            if not ok:
                QMessageBox.information(self, "Error", "Frame rate not selected")
                return
            self.cap = cv2.VideoCapture(file_path)
            self.setup_progress_bar()
            self.process_video_in_thread(file_path, frame_rate)
        else:
            self.cap = cv2.VideoCapture(file_path)
            self.setup_progress_bar()
            self.process_video_in_thread(file_path)

    def setup_progress_bar(self):
        if self.layout() is not None:
            self.clear_layout()

        self.label.setText('Processing...')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedSize(250, 40)

        self.stop_button = QPushButton('Quit', self)
        self.stop_button.setFixedSize(80, 60)
        self.stop_button.clicked.connect(self.stop_detection)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addLayout(progress_layout)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)
        self.progress_bar.setVisible(True)
        self.progress_bar.show()

    def process_video_in_thread(self, file_path, frame_rate=None):
        self.video_thread = VideoProcessingThread(file_path, frame_rate, self.detection_model, self.cap)
        self.video_thread.progress_signal.connect(self.update_progress)
        self.video_thread.finished_signal.connect(self.on_processing_finished)
        self.video_thread.start()

    def on_processing_finished(self):
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Detection Complete", "Video processing is complete.")
        self.init_ui()

    def update_progress(self, progress):
        if self.progress_bar:
            self.progress_bar.setValue(progress)

    def stop_detection(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.terminate() 
        if self.progress_bar:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        self.init_ui()

    def update_frame(self):
        if not self.cap or not self.cap.isOpened() or not hasattr(self, 'video_label') or self.video_label is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        with suppress_stdout():
            results = get_sliced_prediction(
                frame,
                self.detection_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )

        drone_detected_in_frame = False
        for object_prediction in results.object_prediction_list:
            bbox = object_prediction.bbox
            category_name = object_prediction.category.name
            cv2.rectangle(frame, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (0, 255, 0), 2)
            cv2.putText(frame, category_name, (int(bbox.minx), int(bbox.miny) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if category_name.lower() == 'drone':
                drone_detected_in_frame = True

        current_time = datetime.now()

        if drone_detected_in_frame:
            if not self.drone_detected:
                self.drone_detected = True
                self.start_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            self.last_detection_time = current_time
        else:
            if self.drone_detected and self.last_detection_time:
                if (current_time - self.last_detection_time) > timedelta(milliseconds=500):
                    self.drone_detected = False
                    end_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    self.log_timeline(self.start_time, end_time)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convert_to_qt = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt)

        if hasattr(self, 'video_label') and self.video_label is not None:
            self.video_label.setPixmap(pixmap)

            if not self.video_label.isVisible():
                self.video_label.setVisible(True)
                self.resize(rgb_image.shape[1] + 200, rgb_image.shape[0])

    def show_camera_options(self):
        self.clear_layout()
        self.stop_camera()

        main_layout = QVBoxLayout()

        start_button = QPushButton('1. Start Detection', self)
        start_button.clicked.connect(self.setup_camera_ui)
        main_layout.addWidget(start_button)

        timeline_button = QPushButton('2. Show Timeline', self)
        timeline_button.clicked.connect(self.show_timeline_ui)
        main_layout.addWidget(timeline_button)

        back_button = QPushButton('Back to Menu', self) 
        back_button.clicked.connect(self.init_ui)
        main_layout.addWidget(back_button)

        self.setLayout(main_layout)

        self.resize(400, 300)
        self.setFixedSize(400, 300)

    def setup_camera_ui(self):
        self.clear_layout()
        main_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        camera_on_button = QPushButton('Camera On', self)
        camera_on_button.clicked.connect(self.start_camera)

        button_layout.addWidget(camera_on_button)

        camera_off_button = QPushButton('Camera Off', self)
        camera_off_button.clicked.connect(self.stop_camera)
        button_layout.addWidget(camera_off_button)

        main_layout.addLayout(button_layout)


        self.video_label = QLabel(self)
        self.video_label.setVisible(False)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        back_button = QPushButton('Back to Camera Options', self)
        back_button.clicked.connect(self.show_camera_options)
        main_layout.addWidget(back_button)


        self.setLayout(main_layout)

    def show_timeline_ui(self):
        self.clear_layout()

        main_layout = QVBoxLayout()

        label = QLabel('Drone Detection Log')
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(label)

        self.timeline_view = QTextEdit(self)
        self.timeline_view.setReadOnly(True)
        self.timeline_view.setText(self.timeline_content)
        main_layout.addWidget(self.timeline_view)

        reset_timeline_button = QPushButton('Reset Timeline', self)
        reset_timeline_button.clicked.connect(self.reset_timeline)
        main_layout.addWidget(reset_timeline_button)

        back_button = QPushButton('Back to Camera Options', self)
        back_button.clicked.connect(self.show_camera_options)
        main_layout.addWidget(back_button)

        self.load_timeline()

        self.setLayout(main_layout)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215 , 16777215)
        self.resize(800, 800)
        if not self.cap.isOpened():
            QMessageBox.information(self, "Error", "Unable to open camera.")
            return
        self.camera_on = True
        self.timer.start(30)

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.camera_on = False

    def close_application(self):
        self.stop_camera()
        self.close()

    def back_to_main_menu(self):
        self.stop_camera()
        self.init_ui()

    def log_timeline(self, start_time, end_time):
        log_entry = f"{start_time} ------------ {end_time}\n"
        with open(self.log_file, 'a') as log:
            log.write(log_entry)
        self.timeline_content += log_entry

        if hasattr(self, 'timeline_view'):
            self.timeline_view.setText(self.timeline_content)

    def load_timeline(self):
        try:
            with open(self.log_file, 'r') as log:
                self.timeline_content = log.read()
                self.timeline_view.setText(self.timeline_content)
        except FileNotFoundError:
            self.timeline_content = ""
            self.timeline_view.setText(self.timeline_content)

    def reset_timeline(self):
        open(self.log_file, 'w').close()
        self.timeline_content = ""
        if hasattr(self, 'timeline_view'):
            self.timeline_view.setText(self.timeline_content)
        QMessageBox.information(self, "Timeline Reset", "Timeline has been reset.")
    
    # 레이아웃 제거 관련 함수
    def clear_layout(self):
        if self.layout() is not None:
            while self.layout().count():
                item = self.layout().takeAt(0)
                if item.layout():  
                    self.clear_sub_layout(item.layout())  
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)  
                    widget.deleteLater()   
            QWidget().setLayout(self.layout()) 

    def clear_sub_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item.layout(): 
                    self.clear_sub_layout(item.layout())
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DroneDetectionApp()
    ex.show()
    sys.exit(app.exec_())
