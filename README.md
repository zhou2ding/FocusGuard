# Windows11 Focus Timer

一个现代化的 Windows 桌面计时器，支持摄像头监测与语音提醒。

## 功能
- 简约科技感 UI，右下角托盘运行。
- 倒计时悬浮在桌面上，结束后弹窗提醒「计时结束，可以玩手机了。」。
- 计时开始后可开启摄像头监测：检测低头 + 手机出现时语音提示。
- 语音提示内容可自定义。

## 运行
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
python app/main.py
```

macOS 说明：
- 建议使用 Python 3.10/3.11，3.12 可能无法安装 mediapipe。
- Apple Silicon 会使用 `mediapipe-silicon`。

## YOLO11 权重
默认权重路径为 `models/yolo11n.pt`，你可以：
1. 手动下载 YOLO11 权重文件。
2. 放到 `models/` 目录下。
3. 或在界面中选择权重文件路径。

没有权重时仍会计时，但不会触发“玩手机”提示。

## 打包 Windows 可执行文件（在 Windows 上执行）
```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --name FocusTimer app/main.py
```

如果项目较大，也可以在 Windows 上使用 NSIS/Inno Setup 制作安装包。
