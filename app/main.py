import sys
import platform, os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from PySide6.QtGui import QIcon

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

MP_IMPORT_ERROR = ""
try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    MP_IMPORT_ERROR = str(exc)

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
    minutes: int = 20
    seconds: int = 0
    prompt_text: str = "请停止玩手机，专注一下吧。"
    prompt_cooldown: int = 10
    enable_monitor: bool = True
    yolo_weights_path: str = "models/yolo11n.pt"


class FloatingTimer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Window
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFixedSize(240, 90)
        self._drag_offset = None

        self.label = QtWidgets.QLabel("00:00", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont("Arial", 28, QtGui.QFont.Bold)
        self.label.setFont(font)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(self.label)

    def update_time(self, text: str) -> None:
        self.label.setText(text)

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


class VoicePrompter(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self._engine = None
        # Windows 初始化
        if platform.system() == "Windows":
            if pyttsx3:
                try:
                    self._engine = pyttsx3.init()
                except Exception:
                    self._engine = None

    def speak(self, text: str) -> None:
        print(f"[语音调试] 尝试播放: {text}")

        # Mac 走系统命令
        if platform.system() == "Darwin":
            os.system(f'say "{text}" &')
            return

        # Windows 走 pyttsx3
        if self._engine:
            def _worker():
                self._engine.say(text)
                self._engine.runAndWait()
            threading.Thread(target=_worker, daemon=True).start()

class SpinBoxWithButtons(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, minimum: int, maximum: int, value: int, suffix: str):
        super().__init__()
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(value)
        self.spin.setSuffix(suffix)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        self.plus_btn = QtWidgets.QToolButton()
        self.plus_btn.setText("+")
        self.minus_btn = QtWidgets.QToolButton()
        self.minus_btn.setText("-")
        self.plus_btn.clicked.connect(self.spin.stepUp)
        self.minus_btn.clicked.connect(self.spin.stepDown)
        self.spin.valueChanged.connect(self.valueChanged.emit)

        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(2)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.plus_btn)
        btn_layout.addWidget(self.minus_btn)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.spin)
        layout.addLayout(btn_layout)

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, value: int) -> None:
        self.spin.setValue(value)

    def setFixedWidth(self, width: int) -> None:
        self.spin.setFixedWidth(width)


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
        face_mesh_module = None
        if mp:
            if hasattr(mp, "solutions"):
                face_mesh_module = mp.solutions.face_mesh
            else:
                try:
                    from mediapipe import solutions
                    face_mesh_module = solutions.face_mesh
                except Exception:
                    face_mesh_module = None
        if not face_mesh_module:
            try:
                from mediapipe.python.solutions import face_mesh as mp_face_mesh
                face_mesh_module = mp_face_mesh
            except Exception:
                face_mesh_module = None
        if not face_mesh_module:
            self._mp_error = "mediapipe 版本缺少 solutions.face_mesh"
        if not mp:
            self._mp_error = MP_IMPORT_ERROR or "mediapipe 未安装"
        if face_mesh_module:
            self._face_mesh = face_mesh_module.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        weights = Path(self._config.yolo_weights_path)
        if YOLO and weights.exists():
            self._yolo = YOLO(str(weights))
        else:
            self._yolo_missing = True

    def _head_down(self, landmarks) -> bool:
        lm = landmarks.landmark
        eye_left = lm[33]
        eye_right = lm[263]
        nose = lm[1]
        chin = lm[152]
        eye_mid_y = (eye_left.y + eye_right.y) / 2.0
        denom = (chin.y - eye_mid_y)
        if denom <= 0:
            return False
        ratio = (nose.y - eye_mid_y) / denom
        return ratio > 0.45

    def _phone_detected(self, frame) -> bool:
        if not self._yolo:
            return False

        results = self._yolo(frame, conf=0.4, iou=0.5, verbose=False)

        has_phone = False
        objects = []
        for result in results:
            if not hasattr(result, "boxes"):
                continue
            names = result.names
            for cls_id in result.boxes.cls.tolist():
                name = names.get(int(cls_id), "")
                objects.append(name)
                if name in {"cell phone", "phone", "mobile phone", "remote"}:
                    has_phone = True

        return has_phone

    def run(self) -> None:
        self._init_models()
        if not self._face_mesh:
            detail = self._mp_error or "未检测到 mediapipe"
            self.status.emit(f"{detail}，摄像头监测已关闭。")
            return
        if self._yolo_missing:
            self.status.emit("未找到 YOLO11 权重，将仅按低头动作提醒。")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status.emit("无法打开摄像头。")
            return

        last_frame_time = 0.0
        while not self._stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            now = time.time()
            if now - last_frame_time < self._frame_interval:
                continue
            last_frame_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue

            face_landmarks = result.multi_face_landmarks[0]
            head_down = self._head_down(face_landmarks)
            phone = self._phone_detected(frame) if self._yolo else False
            if head_down or phone:
                if now - self._last_prompt_time >= self._config.prompt_cooldown:
                    self._last_prompt_time = now
                    self.prompt_needed.emit(self._config.prompt_text)

        cap.release()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Focus Timer")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "assets", "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"警告: 未找到图标文件 {icon_path}")
        self.setMinimumSize(600, 620)
        self._config = AppConfig()
        self._remaining = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._floating = FloatingTimer()
        self._monitor = None
        self._prompter = VoicePrompter()

        self._build_ui()
        self._apply_style()
        self._init_tray()

    def _build_ui(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 28, 28, 28)

        title = QtWidgets.QLabel("专注计时器")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("倒计时 + 低头玩手机提醒")
        subtitle.setObjectName("subtitle")

        time_group = QtWidgets.QGroupBox("计时设置")
        time_layout = QtWidgets.QFormLayout(time_group)
        time_layout.setHorizontalSpacing(18)
        time_layout.setVerticalSpacing(12)
        time_layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.minutes_input = SpinBoxWithButtons(
            0, 240, self._config.minutes, " 分钟"
        )
        self.minutes_input.setFixedWidth(140)
        self.seconds_input = SpinBoxWithButtons(
            0, 59, self._config.seconds, " 秒"
        )
        self.seconds_input.setFixedWidth(140)
        time_layout.addRow("计时", self.minutes_input)
        time_layout.addRow("", self.seconds_input)

        prompt_group = QtWidgets.QGroupBox("语音提示")
        prompt_layout = QtWidgets.QVBoxLayout(prompt_group)
        prompt_layout.setSpacing(10)
        self.prompt_input = QtWidgets.QLineEdit(self._config.prompt_text)
        self.prompt_input.setPlaceholderText("输入提醒内容")
        self.cooldown_input = SpinBoxWithButtons(
            3, 120, self._config.prompt_cooldown, " 秒"
        )
        self.cooldown_input.setFixedWidth(140)
        cooldown_layout = QtWidgets.QHBoxLayout()
        cooldown_layout.setSpacing(12)
        cooldown_layout.addWidget(QtWidgets.QLabel("提醒冷却(秒)"))
        cooldown_layout.addWidget(self.cooldown_input)
        prompt_layout.addWidget(self.prompt_input)
        prompt_layout.addLayout(cooldown_layout)

        monitor_group = QtWidgets.QGroupBox("摄像头监测")
        monitor_layout = QtWidgets.QVBoxLayout(monitor_group)
        monitor_layout.setSpacing(10)
        self.monitor_checkbox = QtWidgets.QCheckBox("开始计时后启用监测")
        self.monitor_checkbox.setChecked(self._config.enable_monitor)
        self.model_path = QtWidgets.QLineEdit(self._config.yolo_weights_path)
        self.model_path.setPlaceholderText("YOLO11 权重路径 (可选)")
        self.browse_button = QtWidgets.QPushButton("选择权重")
        self.browse_button.clicked.connect(self._browse_weights)
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.setSpacing(12)
        path_layout.addWidget(self.model_path)
        path_layout.addWidget(self.browse_button)
        monitor_layout.addWidget(self.monitor_checkbox)
        monitor_layout.addLayout(path_layout)
        self.monitor_status = QtWidgets.QLabel("")
        self.monitor_status.setObjectName("status")
        monitor_layout.addWidget(self.monitor_status)

        self.start_button = QtWidgets.QPushButton("开始计时")
        self.start_button.clicked.connect(self._on_start)
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(self._on_stop)
        self.stop_button.setEnabled(False)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(time_group)
        layout.addWidget(prompt_group)
        layout.addWidget(monitor_group)
        layout.addStretch()
        layout.addLayout(button_layout)

        self.setCentralWidget(widget)

    def _apply_style(self):
        font_family = "Microsoft YaHei UI"
        if sys.platform == "darwin":
            font_family = "PingFang SC"
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: #fff6ef;
                color: #3d2b1f;
                font-family: "{font_family}";
                font-size: 14px;
            }}
            #title {{
                font-size: 28px;
                font-weight: 700;
                color: #3d2b1f;
            }}
            #subtitle {{
                color: #7b5e4a;
                margin-bottom: 10px;
            }}
            QGroupBox {{
                border: 1px solid #f0c6a8;
                border-radius: 14px;
                margin-top: 16px;
                padding-top: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px 0 6px;
                color: #9a6b4b;
            }}
            QLineEdit, QSpinBox {{
                background-color: #fff1e8;
                border: 1px solid #f2b48f;
                border-radius: 10px;
                padding: 2px 10px;
                min-height: 40px;
                selection-background-color: #f08a5d;
            }}
            QToolButton {{
                background-color: #ffe4d6;
                border: 1px solid #f2b48f;
                border-radius: 6px;
                min-width: 20px;
                min-height: 16px;
                font-weight: 700;
                color: #8a5a3a;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 16px;
                border-radius: 6px;
                background-color: #ffe4d6;
                border: 1px solid #f2b48f;
                subcontrol-origin: padding;
            }}
            QSpinBox::up-button {{
                subcontrol-position: top right;
                margin: 5px 4px 0 0;
            }}
            QSpinBox::down-button {{
                subcontrol-position: bottom right;
                margin: 0 4px 5px 0;
            }}
            QSpinBox::up-arrow, QSpinBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QCheckBox {{
                spacing: 6px;
                min-height: 24px;
            }}
            QPushButton {{
                background-color: #f08a5d;
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
                color: #fff6ef;
                min-height: 34px;
            }}
            QPushButton:disabled {{
                background-color: #f3c7ae;
                color: #b78970;
            }}
            #status {{
                color: #b06b4b;
            }}
            """
        )

    def _init_tray(self):
        self.tray = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        self.tray.setIcon(icon)
        self.tray.setToolTip("Focus Timer")

        menu = QtWidgets.QMenu()
        action_show = menu.addAction("显示/隐藏")
        action_show.triggered.connect(self._toggle_visibility)
        action_quit = menu.addAction("退出")
        action_quit.triggered.connect(self._on_quit)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.showNormal()
            self.raise_()
            self.activateWindow()

    def _browse_weights(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", "", "Model Files (*.pt)"
        )
        if path:
            self.model_path.setText(path)

    def _on_start(self):
        minutes = self.minutes_input.value()
        seconds = self.seconds_input.value()
        total = minutes * 60 + seconds
        if total <= 0:
            QtWidgets.QMessageBox.information(self, "提示", "请输入有效计时时长。")
            return

        self._config.minutes = minutes
        self._config.seconds = seconds
        self._config.prompt_text = self.prompt_input.text().strip() or self._config.prompt_text
        self._config.prompt_cooldown = self.cooldown_input.value()
        self._config.enable_monitor = self.monitor_checkbox.isChecked()
        self._config.yolo_weights_path = self.model_path.text().strip()

        self._remaining = total
        self._update_countdown_label()
        self._timer.start(1000)
        self._floating.show()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        if self._config.enable_monitor:
            self._start_monitor()

    def _start_monitor(self):
        self.monitor_status.setText("摄像头监测已启动")
        self._monitor = CameraMonitor(self._config)
        self._monitor.prompt_needed.connect(self._on_prompt)
        self._monitor.status.connect(self._on_monitor_status)
        self._monitor.start()

    def _on_prompt(self, text: str):
        self._prompter.speak(text)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text)

    def _on_monitor_status(self, message: str):
        self.monitor_status.setText(message)

    def _on_stop(self):
        self._timer.stop()
        self._floating.hide()
        self._stop_monitor()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_quit(self):
        self._on_stop()
        QtWidgets.QApplication.quit()

    def _stop_monitor(self):
        if self._monitor:
            self._monitor.stop()
            self._monitor.wait(1500)
            self._monitor = None

    def _on_tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self._timer.stop()
            self._floating.hide()
            self._stop_monitor()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            end_text = "计时结束，可以玩手机了。"
            self._prompter.speak(end_text)
            QtWidgets.QMessageBox.information(self, "提示", end_text)
            return
        self._update_countdown_label()

    def _update_countdown_label(self):
        mins = self._remaining // 60
        secs = self._remaining % 60
        text = f"{mins:02d}:{secs:02d}"
        self._floating.update_time(text)

    def closeEvent(self, event):
        event.ignore()
        self.hide()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
