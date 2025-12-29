import sys
import os
import time
import threading
import queue
import signal
from dataclasses import dataclass
from pathlib import Path

# --- 界面库 ---
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QIcon, QAction, QImage, QPixmap


# --- 屏蔽烦人的 Qt 日志 ---
def qt_message_handler(mode, context, message):
    if "QFont::setPointSize" in message:
        return  # 忽略这条特定的日志
    # 打印其他日志 (可选)
    # print(f"[Qt Log] {message}")


QtCore.qInstallMessageHandler(qt_message_handler)

# --- 视觉库 ---
import cv2

# --- 依赖库导入检查 ---
MP_IMPORT_ERROR = ""
mp_face_mesh = None

try:
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        mp_face_mesh = mp.solutions.face_mesh
    else:
        MP_IMPORT_ERROR = "MediaPipe加载不完整"
except ImportError as e:
    MP_IMPORT_ERROR = f"未安装 MediaPipe: {e}"
except Exception as e:
    MP_IMPORT_ERROR = f"MediaPipe 内部错误: {e}"

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


@dataclass
class AppConfig:
    focus_minutes: int = 25
    focus_seconds: int = 0
    break_minutes: int = 5
    break_seconds: int = 0
    prompt_text: str = "请停止玩手机，专注一下吧。"
    prompt_cooldown: int = 8
    enable_monitor: bool = True
    yolo_weights_path: str = "models/yolo11n.pt"
    camera_index: int = 0


# --- 悬浮窗组件 ---
class FloatingTimer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setFixedSize(240, 90)
        self._drag_offset = None

        self.label = QtWidgets.QLabel("00:00", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont("Arial", 28, QtGui.QFont.Bold)
        self.label.setFont(font)

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setGeometry(0, 65, 240, 20)
        self.status_label.setStyleSheet("color: #8a5a3a; font-size: 12px; font-weight: bold;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(self.label)

    def update_time(self, text: str, status_text: str = "") -> None:
        self.label.setText(text)
        if status_text:
            self.status_label.setText(status_text)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QtGui.QColor(246, 234, 222, 220))
        gradient.setColorAt(1.0, QtGui.QColor(228, 206, 192, 220))
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtGui.QPen(QtGui.QColor(186, 128, 98, 180), 1.5))
        painter.drawRoundedRect(rect, 16, 16)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self._drag_offset:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        event.accept()


# --- 语音组件 (暴力重置版) ---
class VoicePrompter(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        print(f"[语音指令] 加入队列: {text}")
        self._queue.put(text)

    def _speech_loop(self):
        while True:
            text = self._queue.get()
            if text is None: break

            # 【核心修改】每次说话都重新初始化引擎
            # 这样可以避免上一句的 COM 对象死锁影响下一句
            if pyttsx3:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(text)
                    engine.runAndWait()
                    # 显式停止，释放资源
                    engine.stop()
                    del engine
                except Exception as e:
                    print(f"[语音错误] 播放失败: {e}")
            else:
                print(f"[语音模拟] {text}")


# --- 自定义输入框 ---
class SpinBoxWithButtons(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, minimum: int, maximum: int, value: int, suffix: str):
        super().__init__()
        self.setMinimumHeight(40)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(value)
        self.spin.setSuffix(suffix)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.plus_btn = QtWidgets.QToolButton()
        self.plus_btn.setText("+")
        self.minus_btn = QtWidgets.QToolButton()
        self.minus_btn.setText("-")
        self.plus_btn.setFixedSize(20, 18)
        self.minus_btn.setFixedSize(20, 18)
        self.plus_btn.clicked.connect(self.spin.stepUp)
        self.minus_btn.clicked.connect(self.spin.stepDown)
        self.spin.valueChanged.connect(self.valueChanged.emit)
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(0)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.plus_btn)
        btn_layout.addWidget(self.minus_btn)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spin)
        layout.addLayout(btn_layout)

    def value(self) -> int: return self.spin.value()

    def setValue(self, value: int) -> None: self.spin.setValue(value)


# --- 监测线程 ---
class CameraMonitor(QtCore.QThread):
    prompt_needed = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self._config = config
        self._stop_flag = threading.Event()
        self._last_prompt_time = 0.0
        self._face_mesh = None
        self._yolo = None
        self._mp_error = None
        self.HEAD_DOWN_THRESHOLD = 0.45

    def stop(self) -> None:
        self._stop_flag.set()

    def _init_models(self) -> None:
        if mp_face_mesh:
            try:
                self._face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                self._mp_error = str(e)
        else:
            self._mp_error = MP_IMPORT_ERROR or "Mediapipe库未加载"

        weights = Path(self._config.yolo_weights_path)
        if YOLO and weights.exists():
            try:
                self._yolo = YOLO(str(weights))
            except Exception:
                pass

    def _calculate_ratio(self, landmarks):
        lm = landmarks.landmark
        nose = lm[1]
        chin = lm[152]
        eye_mid_y = (lm[33].y + lm[263].y) / 2.0
        denom = (chin.y - eye_mid_y)
        if denom <= 0: return 0.0
        return (nose.y - eye_mid_y) / denom

    def _phone_detected(self, frame) -> bool:
        if not self._yolo: return False
        results = self._yolo(frame, conf=0.4, iou=0.5, verbose=False)
        for result in results:
            if not hasattr(result, "boxes"): continue
            names = result.names
            for cls_id in result.boxes.cls.tolist():
                name = names.get(int(cls_id), "")
                if name in {"cell phone", "phone", "mobile phone"}:
                    return True
        return False

    def run(self) -> None:
        self._init_models()
        if not self._face_mesh:
            self.status.emit(f"错误: {self._mp_error}")
            return

        cap = cv2.VideoCapture(self._config.camera_index)
        if not cap.isOpened():
            self.status.emit(f"无法打开摄像头 {self._config.camera_index}")
            return

        print(f"[监测线程] 已启动，使用摄像头 {self._config.camera_index}")
        last_frame_time = 0.0

        while not self._stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_frame_time < 0.2:
                continue
            last_frame_time = now

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._face_mesh.process(rgb)

                is_head_down = False
                has_phone = False

                if res.multi_face_landmarks:
                    ratio = self._calculate_ratio(res.multi_face_landmarks[0])
                    if ratio > self.HEAD_DOWN_THRESHOLD:
                        is_head_down = True

                if self._yolo and (not is_head_down):
                    if self._phone_detected(frame):
                        has_phone = True

                trigger = is_head_down or has_phone

                if trigger:
                    if now - self._last_prompt_time >= self._config.prompt_cooldown:
                        self._last_prompt_time = now
                        print(f"[触发警告] 低头: {is_head_down}, 手机: {has_phone}")
                        self.prompt_needed.emit(self._config.prompt_text)

            except Exception:
                pass

        cap.release()
        print("[监测线程] 已退出")


# --- 主窗口 ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeadsUp - 专注卫士")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "assets", "logo.png")
        if os.path.exists(icon_path):
            self._app_icon = QIcon(icon_path)
            self.setWindowIcon(self._app_icon)
            QtWidgets.QApplication.setWindowIcon(self._app_icon)

        self.setMinimumSize(600, 800)
        self._config = AppConfig()

        self._remaining = 0
        self._is_break_mode = False

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._floating = FloatingTimer()
        self._monitor = None
        self._prompter = VoicePrompter()

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_cap = None
        self._is_previewing = False
        self._camera_verified = False

        self._build_ui()
        self._apply_style()
        self._init_tray()

        self._scan_cameras()

    def _scan_cameras(self):
        self.camera_combo.clear()
        found = False
        for i in range(3):
            temp = cv2.VideoCapture(i)
            if temp.isOpened():
                ret, _ = temp.read()
                if ret:
                    self.camera_combo.addItem(f"摄像头 {i}", i)
                    found = True
            temp.release()

        if not found:
            self.camera_combo.addItem("未检测到摄像头", -1)
            self.preview_btn.setEnabled(False)
        else:
            self.camera_combo.setCurrentIndex(0)

    def _toggle_preview(self):
        if self._is_previewing:
            self._stop_preview()
            self.preview_btn.setText("测试/预览摄像头")
        else:
            idx = self.camera_combo.currentData()
            if idx is None or idx < 0:
                QtWidgets.QMessageBox.warning(self, "错误", "没有可用的摄像头")
                return

            self._preview_cap = cv2.VideoCapture(idx)
            if self._preview_cap.isOpened():
                self._is_previewing = True
                self._preview_timer.start(30)
                self.preview_btn.setText("停止预览")
                self.monitor_status.setText("正在预览画面...")

                self._camera_verified = True
                self.start_button.setEnabled(True)
                self.start_button.setText("开始专注循环")
                self.start_button.setStyleSheet("""
                    background-color: #28a745; 
                    color: white; 
                    font-weight: bold;
                    border-radius: 10px;
                """)
            else:
                QtWidgets.QMessageBox.critical(self, "错误", "无法打开该摄像头，请选择其他设备。")

    def _update_preview(self):
        if self._preview_cap and self._preview_cap.isOpened():
            ret, frame = self._preview_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img).scaled(
                    self.preview_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(pixmap)
            else:
                self.monitor_status.setText("摄像头读取失败")

    def _stop_preview(self):
        self._preview_timer.stop()
        if self._preview_cap:
            self._preview_cap.release()
            self._preview_cap = None
        self._is_previewing = False
        self.preview_label.clear()
        self.preview_label.setText("摄像头预览区\n(请点击下方按钮测试)")

    def _browse_weights(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型", "", "Model (*.pt)"
        )
        if path:
            self.model_path_input.setText(path)

    def _on_start(self):
        if self._is_previewing:
            self._stop_preview()
            self.preview_btn.setText("测试/预览摄像头")

        self._config.focus_minutes = self.focus_min_input.value()
        self._config.focus_seconds = self.focus_sec_input.value()
        self._config.break_minutes = self.break_min_input.value()
        self._config.break_seconds = self.break_sec_input.value()
        self._config.prompt_text = self.prompt_input.text()
        self._config.camera_index = self.camera_combo.currentData()
        self._config.yolo_weights_path = self.model_path_input.text().strip()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.preview_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.model_path_input.setEnabled(False)
        self.browse_btn.setEnabled(False)

        self._floating.show()
        self._start_focus_phase()
        self._timer.start(1000)

    def _on_stop(self):
        self._timer.stop()
        self._floating.hide()
        self._stop_monitor_cleanly()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.model_path_input.setEnabled(True)
        self.browse_btn.setEnabled(True)

        self.monitor_status.setText("已停止")

    def _start_focus_phase(self):
        self._is_break_mode = False
        self._remaining = self._config.focus_minutes * 60 + self._config.focus_seconds
        self._update_countdown_label()
        self._prompter.speak("开始专注")

        if self._config.enable_monitor:
            self._monitor = CameraMonitor(self._config)
            self._monitor.prompt_needed.connect(lambda t: self._prompter.speak(t))
            self._monitor.status.connect(lambda s: self.monitor_status.setText(s))
            self._monitor.start()
            self.monitor_status.setText("AI监测运行中...")
        else:
            self.monitor_status.setText("监测未开启")

    def _start_break_phase(self):
        self._is_break_mode = True
        self._remaining = self._config.break_minutes * 60 + self._config.break_seconds
        self._update_countdown_label()
        self._prompter.speak("休息时间")
        self._stop_monitor_cleanly()
        self.monitor_status.setText("休息中 (监测暂停)")

    def _stop_monitor_cleanly(self):
        if self._monitor:
            self._monitor.stop()
            self._monitor.wait(500)
            self._monitor = None

    def _on_tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            if not self._is_break_mode:
                self._start_break_phase()
            else:
                self._start_focus_phase()
        else:
            self._update_countdown_label()

    def _update_countdown_label(self):
        m, s = divmod(self._remaining, 60)
        self._floating.update_time(f"{m:02d}:{s:02d}", "休息" if self._is_break_mode else "专注")

    def _init_tray(self):
        self.tray = QtWidgets.QSystemTrayIcon(self)
        if hasattr(self, "_app_icon"): self.tray.setIcon(self._app_icon)

        menu = QtWidgets.QMenu()
        action_show = QAction("显示主界面", self)
        action_show.triggered.connect(self._show_window)
        menu.addAction(action_show)
        menu.addSeparator()
        action_quit = QAction("退出程序", self)
        action_quit.triggered.connect(self._on_force_quit)
        menu.addAction(action_quit)

        self.tray.setContextMenu(menu)
        self.tray.show()
        self.tray.activated.connect(self._on_tray_activated)

    def _on_tray_activated(self, reason):
        if reason in (QtWidgets.QSystemTrayIcon.Trigger, QtWidgets.QSystemTrayIcon.DoubleClick):
            self._show_window()

    def _show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_force_quit(self):
        self._stop_monitor_cleanly()
        if self._is_previewing:
            self._stop_preview()
        self.tray.hide()
        os._exit(0)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray.showMessage("HeadsUp", "程序已最小化到托盘", QtWidgets.QSystemTrayIcon.Information, 1000)

    def _build_ui(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QtWidgets.QLabel("HeadsUp 专注卫士")
        title.setObjectName("title")

        # --- 1. 设备设置 ---
        cam_group = QtWidgets.QGroupBox("1. 硬件检查 (必须完成)")
        cam_layout = QtWidgets.QVBoxLayout(cam_group)

        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.setMinimumHeight(35)

        self.preview_label = QtWidgets.QLabel("摄像头预览区\n(请点击下方按钮测试)")
        self.preview_label.setFixedSize(320, 240)
        self.preview_label.setStyleSheet("background-color: #333; color: #fff; border-radius: 5px;")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)

        self.preview_btn = QtWidgets.QPushButton("测试/预览摄像头")
        self.preview_btn.clicked.connect(self._toggle_preview)

        preview_container = QtWidgets.QHBoxLayout()
        preview_container.addStretch()
        preview_container.addWidget(self.preview_label)
        preview_container.addStretch()

        model_layout = QtWidgets.QHBoxLayout()
        self.model_path_input = QtWidgets.QLineEdit(self._config.yolo_weights_path)
        self.model_path_input.setPlaceholderText("YOLO模型路径 (.pt)")
        self.browse_btn = QtWidgets.QPushButton("...")
        self.browse_btn.setFixedWidth(40)
        self.browse_btn.clicked.connect(self._browse_weights)
        model_layout.addWidget(QtWidgets.QLabel("模型:"))
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(self.browse_btn)

        cam_layout.addWidget(QtWidgets.QLabel("选择摄像头:"))
        cam_layout.addWidget(self.camera_combo)
        cam_layout.addLayout(preview_container)
        cam_layout.addWidget(self.preview_btn)
        cam_layout.addSpacing(10)
        cam_layout.addLayout(model_layout)

        # --- 2. 时间设置 ---
        time_group = QtWidgets.QGroupBox("2. 专注计划")
        time_layout = QtWidgets.QGridLayout(time_group)
        self.focus_min_input = SpinBoxWithButtons(0, 240, self._config.focus_minutes, " 分")
        self.focus_sec_input = SpinBoxWithButtons(0, 59, self._config.focus_seconds, " 秒")
        self.break_min_input = SpinBoxWithButtons(0, 60, self._config.break_minutes, " 分")
        self.break_sec_input = SpinBoxWithButtons(0, 59, self._config.break_seconds, " 秒")
        time_layout.addWidget(QtWidgets.QLabel("专注:"), 0, 0)
        time_layout.addWidget(self.focus_min_input, 0, 1)
        time_layout.addWidget(self.focus_sec_input, 0, 2)
        time_layout.addWidget(QtWidgets.QLabel("休息:"), 1, 0)
        time_layout.addWidget(self.break_min_input, 1, 1)
        time_layout.addWidget(self.break_sec_input, 1, 2)

        # --- 3. 提醒 ---
        prompt_group = QtWidgets.QGroupBox("3. 提醒内容")
        prompt_layout = QtWidgets.QVBoxLayout(prompt_group)
        self.prompt_input = QtWidgets.QLineEdit(self._config.prompt_text)
        prompt_layout.addWidget(self.prompt_input)

        self.monitor_status = QtWidgets.QLabel("等待摄像头检测...")
        self.monitor_status.setObjectName("status")

        self.start_button = QtWidgets.QPushButton("请先测试摄像头")
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("background-color: #cccccc; color: #666;")
        self.start_button.clicked.connect(self._on_start)

        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(self._on_stop)
        self.stop_button.setEnabled(False)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.stop_button)

        layout.addWidget(title)
        layout.addWidget(cam_group)
        layout.addWidget(time_group)
        layout.addWidget(prompt_group)
        layout.addWidget(self.monitor_status)
        layout.addLayout(btn_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #fff6ef; color: #3d2b1f; font-family: "Microsoft YaHei UI"; font-size: 14px; }
            #title { font-size: 22px; font-weight: bold; margin-bottom: 10px; }
            QGroupBox { border: 1px solid #f0c6a8; border-radius: 8px; margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QLineEdit, QSpinBox, QComboBox { background: #fff1e8; border: 1px solid #f2b48f; border-radius: 5px; min-height: 30px; padding: 0 5px; }
            QPushButton { background: #f08a5d; color: white; border-radius: 5px; min-height: 38px; font-weight: bold; }
            QPushButton:disabled { background: #e0c0b0; color: #fff; }
            #status { color: #d35400; font-weight: bold; margin-top: 5px; }
        """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())