import sys
import platform, os
import time
import threading
import signal
from dataclasses import dataclass
from pathlib import Path

# --- 界面库 ---
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QIcon, QAction

# --- 视觉库导入 (带防崩溃处理) ---
import cv2

MP_IMPORT_ERROR = ""
mp_face_mesh = None

try:
    # 【核心修复】 回归最标准的导入方式
    # 只要完成了"第一步"的环境修复，这里就能正常工作
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        mp_face_mesh = mp.solutions.face_mesh
    else:
        # 如果 import mediapipe 成功但没有 solutions，通常是 protobuf 版本不对
        MP_IMPORT_ERROR = "MediaPipe加载不完整，请检查 protobuf 版本"
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


# --- 配置数据类 ---
@dataclass
class AppConfig:
    focus_minutes: int = 25
    focus_seconds: int = 0
    break_minutes: int = 5
    break_seconds: int = 0
    prompt_text: str = "请停止玩手机，专注一下吧。"
    prompt_cooldown: int = 10
    enable_monitor: bool = True
    yolo_weights_path: str = "models/yolo11n.pt"


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


# --- 语音播报组件 ---
class VoicePrompter(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self._engine = None
        if platform.system() == "Windows":
            if pyttsx3:
                try:
                    self._engine = pyttsx3.init()
                except Exception:
                    self._engine = None

    def speak(self, text: str) -> None:
        if platform.system() == "Darwin":
            os.system(f'say "{text}" &')
            return

        if self._engine:
            def _worker():
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except:
                    pass

            threading.Thread(target=_worker, daemon=True).start()


# --- 带加减按钮的输入框 ---
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

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, value: int) -> None:
        self.spin.setValue(value)


# --- 摄像头监测线程 ---
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
        self._yolo_missing = False
        self._frame_interval = 0.3
        self._mp_error = None

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
            self._mp_error = MP_IMPORT_ERROR or "Mediapipe 加载失败"

        weights = Path(self._config.yolo_weights_path)
        if YOLO and weights.exists():
            try:
                self._yolo = YOLO(str(weights))
            except Exception:
                self._yolo_missing = True
        else:
            self._yolo_missing = True

    def _head_down(self, landmarks) -> bool:
        lm = landmarks.landmark
        nose = lm[1]
        chin = lm[152]
        # 使用眼睛的平均高度
        eye_mid_y = (lm[33].y + lm[263].y) / 2.0
        denom = (chin.y - eye_mid_y)
        if denom <= 0: return False
        ratio = (nose.y - eye_mid_y) / denom
        return ratio > 0.45

    def _phone_detected(self, frame) -> bool:
        if not self._yolo: return False
        results = self._yolo(frame, conf=0.4, iou=0.5, verbose=False)
        for result in results:
            if not hasattr(result, "boxes"): continue
            names = result.names
            for cls_id in result.boxes.cls.tolist():
                if names.get(int(cls_id), "") in {"cell phone", "phone", "mobile phone", "remote"}:
                    return True
        return False

    def run(self) -> None:
        self._init_models()
        if not self._face_mesh:
            self.status.emit(f"错误: {self._mp_error}")
            return

        cap = None
        # 尝试摄像头索引
        for idx in ([1, 0] if sys.platform == "darwin" else [0, 1]):
            if self._stop_flag.is_set(): break
            temp = cv2.VideoCapture(idx)
            if temp.isOpened():
                ret, _ = temp.read()
                if ret:
                    cap = temp
                    break
                temp.release()

        if not cap:
            self.status.emit("无法打开摄像头")
            return

        last_frame_time = 0.0
        while not self._stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_frame_time < self._frame_interval:
                time.sleep(0.05)
                continue
            last_frame_time = now

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._face_mesh.process(rgb)

                trigger = False
                # 1. 检测低头
                if res.multi_face_landmarks:
                    if self._head_down(res.multi_face_landmarks[0]):
                        trigger = True

                # 2. 检测手机
                if not trigger and self._yolo:
                    if self._phone_detected(frame):
                        trigger = True

                if trigger:
                    if now - self._last_prompt_time >= self._config.prompt_cooldown:
                        self._last_prompt_time = now
                        self.prompt_needed.emit(self._config.prompt_text)
            except Exception:
                pass

        cap.release()


# --- 主窗口 ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeadsUp - 专注卫士")

        # 允许 Ctrl+C 强制结束
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # 标志位：是否强制退出
        self._force_quit = False

        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "assets", "logo.png")
        if os.path.exists(icon_path):
            self._app_icon = QIcon(icon_path)
        else:
            self._app_icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)

        self.setWindowIcon(self._app_icon)
        QtWidgets.QApplication.setWindowIcon(self._app_icon)

        self.setMinimumSize(600, 680)
        self._config = AppConfig()

        self._remaining = 0
        self._is_break_mode = False

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._floating = FloatingTimer()
        self._monitor = None
        self._prompter = VoicePrompter()

        self._build_ui()
        self._apply_style()
        self._init_tray()

    def _init_tray(self):
        self.tray = QtWidgets.QSystemTrayIcon(self)
        self.tray.setIcon(self._app_icon)
        self.tray.setToolTip("HeadsUp 专注卫士")

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
        if reason == QtWidgets.QSystemTrayIcon.Trigger or reason == QtWidgets.QSystemTrayIcon.DoubleClick:
            self._show_window()

    def _show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_force_quit(self):
        self._force_quit = True
        self.close()

    def closeEvent(self, event):
        if self._force_quit:
            self._stop_monitor_cleanly()
            self._timer.stop()
            self.tray.hide()
            event.accept()
        else:
            event.ignore()
            self.hide()
            self.tray.showMessage(
                "HeadsUp 仍在运行",
                "程序已最小化到托盘，监测仍在继续。",
                QtWidgets.QSystemTrayIcon.Information,
                2000
            )

    def _stop_monitor_cleanly(self):
        if self._monitor:
            self._monitor.stop()
            self._monitor.wait(1000)
            self._monitor = None

    def _build_ui(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 28, 28, 28)

        title = QtWidgets.QLabel("HeadsUp 专注卫士")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("窗口关闭后将最小化到托盘，保持后台监测")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)

        time_group = QtWidgets.QGroupBox("时间循环设置")
        time_layout = QtWidgets.QGridLayout(time_group)
        time_layout.setHorizontalSpacing(18)
        time_layout.setVerticalSpacing(12)

        self.focus_min_input = SpinBoxWithButtons(0, 240, self._config.focus_minutes, " 分钟")
        self.focus_min_input.setFixedWidth(150)
        self.focus_sec_input = SpinBoxWithButtons(0, 59, self._config.focus_seconds, " 秒")
        self.focus_sec_input.setFixedWidth(150)
        self.break_min_input = SpinBoxWithButtons(0, 60, self._config.break_minutes, " 分钟")
        self.break_min_input.setFixedWidth(150)
        self.break_sec_input = SpinBoxWithButtons(0, 59, self._config.break_seconds, " 秒")
        self.break_sec_input.setFixedWidth(150)

        time_layout.addWidget(QtWidgets.QLabel("专注时长"), 0, 0)
        time_layout.addWidget(self.focus_min_input, 0, 1)
        time_layout.addWidget(self.focus_sec_input, 0, 2)
        time_layout.addWidget(QtWidgets.QLabel("休息时长"), 1, 0)
        time_layout.addWidget(self.break_min_input, 1, 1)
        time_layout.addWidget(self.break_sec_input, 1, 2)
        time_layout.setColumnStretch(3, 1)

        prompt_group = QtWidgets.QGroupBox("提醒设置")
        prompt_layout = QtWidgets.QVBoxLayout(prompt_group)
        self.prompt_input = QtWidgets.QLineEdit(self._config.prompt_text)
        self.cooldown_input = SpinBoxWithButtons(3, 120, self._config.prompt_cooldown, " 秒")
        self.cooldown_input.setFixedWidth(150)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("冷却间隔"))
        row.addWidget(self.cooldown_input)
        row.addStretch()
        prompt_layout.addWidget(self.prompt_input)
        prompt_layout.addLayout(row)

        monitor_group = QtWidgets.QGroupBox("摄像头设置")
        monitor_layout = QtWidgets.QVBoxLayout(monitor_group)
        self.monitor_checkbox = QtWidgets.QCheckBox("启用监测")
        self.monitor_checkbox.setChecked(self._config.enable_monitor)
        self.model_path = QtWidgets.QLineEdit(self._config.yolo_weights_path)
        self.browse_button = QtWidgets.QPushButton("...")
        self.browse_button.setFixedWidth(40)
        self.browse_button.clicked.connect(self._browse_weights)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.model_path)
        path_row.addWidget(self.browse_button)
        monitor_layout.addWidget(self.monitor_checkbox)
        monitor_layout.addLayout(path_row)
        self.monitor_status = QtWidgets.QLabel("就绪")
        self.monitor_status.setObjectName("status")
        monitor_layout.addWidget(self.monitor_status)

        self.start_button = QtWidgets.QPushButton("开始")
        self.start_button.clicked.connect(self._on_start)
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(self._on_stop)
        self.stop_button.setEnabled(False)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.stop_button)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(time_group)
        layout.addWidget(prompt_group)
        layout.addWidget(monitor_group)
        layout.addStretch()
        layout.addLayout(btn_row)
        self.setCentralWidget(widget)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #fff6ef; color: #3d2b1f; font-family: "Microsoft YaHei UI"; font-size: 14px; }
            #title { font-size: 24px; font-weight: bold; }
            #subtitle { color: #888; font-size: 12px; margin-bottom: 10px; }
            QGroupBox { border: 1px solid #f0c6a8; border-radius: 8px; margin-top: 10px; padding-top: 10px; }
            QLineEdit, QSpinBox { background: #fff1e8; border: 1px solid #f2b48f; border-radius: 5px; min-height: 35px; }
            QPushButton { background: #f08a5d; color: white; border-radius: 5px; min-height: 35px; font-weight: bold; }
            QPushButton:disabled { background: #e0c0b0; }
        """)

    def _browse_weights(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型", "", "Model (*.pt)")
        if path: self.model_path.setText(path)

    def _on_start(self):
        self._config.focus_minutes = self.focus_min_input.value()
        self._config.focus_seconds = self.focus_sec_input.value()
        self._config.break_minutes = self.break_min_input.value()
        self._config.break_seconds = self.break_sec_input.value()
        self._config.prompt_text = self.prompt_input.text()
        self._config.prompt_cooldown = self.cooldown_input.value()
        self._config.enable_monitor = self.monitor_checkbox.isChecked()
        self._config.yolo_weights_path = self.model_path.text()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._floating.show()
        self._start_focus_phase()
        self._timer.start(1000)

    def _on_stop(self):
        self._timer.stop()
        self._floating.hide()
        self._stop_monitor_cleanly()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
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
            self.monitor_status.setText("监测开启")
        else:
            self.monitor_status.setText("监测未开启")

    def _start_break_phase(self):
        self._is_break_mode = True
        self._remaining = self._config.break_minutes * 60 + self._config.break_seconds
        self._update_countdown_label()
        self._prompter.speak("休息时间")
        self._stop_monitor_cleanly()
        self.monitor_status.setText("休息中 (监测暂停)")

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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())