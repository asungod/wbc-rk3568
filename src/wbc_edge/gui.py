import argparse
import sys
import cv2
from PySide6.QtCore import QThread, QTimer, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from .config import load_config
from .pipeline import Pipeline, annotate, save_sample
from .upload import upload_sample


class Task(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg, image=None, folder=None, parent=None):
        super().__init__(parent)
        self.cfg, self.image, self.folder = cfg, image, folder

    def run(self):
        try:
            if self.folder is not None:
                status = upload_sample(self.folder, self.cfg["upload"])
                self.completed.emit({"http": status})
                return
            pipe = Pipeline(self.cfg)
            try:
                result = pipe.analyze(self.image)
            finally:
                pipe.close()
            folder = save_sample(self.image, result, self.cfg["output_dir"])
            self.completed.emit({"result": result, "folder": str(folder),
                                 "image": annotate(self.image, result)})
        except Exception as exc:
            self.failed.emit(str(exc))


class Window(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.cap, self.frame, self.folder, self.task = cfg, None, None, None, None
        self.setWindowTitle("白细胞端侧分析")
        self.resize(1024, 700)
        self.preview = QLabel("连接摄像头或打开图片")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 400)
        self.info = QLabel("待处理")
        self.info.setWordWrap(True)
        self.camera_button = QPushButton("连接摄像头")
        self.file_button = QPushButton("打开图片")
        self.analyze_button = QPushButton("拍照并分析")
        self.upload_button = QPushButton("上传待复核样本")
        self.upload_button.setEnabled(False)
        self.camera_button.clicked.connect(self.open_camera)
        self.file_button.clicked.connect(self.open_image)
        self.analyze_button.clicked.connect(self.analyze)
        self.upload_button.clicked.connect(self.upload)
        row = QHBoxLayout()
        for button in (self.camera_button, self.file_button, self.analyze_button, self.upload_button):
            row.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.preview)
        layout.addWidget(self.info)
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_camera)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(self.cfg["camera"]["index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["camera"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["camera"]["height"])
        if not self.cap.isOpened():
            self.stop_camera()
            self.info.setText("摄像头打开失败")
            return
        self.folder = None
        self.upload_button.setEnabled(False)
        self.timer.start(50)
        self.info.setText("摄像头预览中")

    def read_camera(self):
        ok, frame = self.cap.read()
        if not ok:
            self.stop_camera()
            self.info.setText("摄像头读取失败")
            return
        self.frame = frame
        self.show_image(frame)

    def show_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            self.info.setText("无法读取图片")
            return
        self.stop_camera()
        self.frame, self.folder = image, None
        self.upload_button.setEnabled(False)
        self.show_image(image)
        self.info.setText("图片已加载")

    def start_task(self, task):
        self.task = task
        for b in (self.file_button, self.camera_button, self.analyze_button, self.upload_button):
            b.setEnabled(False)
        task.completed.connect(self.finished_result)
        task.failed.connect(lambda error: self.info.setText(f"操作失败：{error}"))
        task.finished.connect(self.unlock)
        task.finished.connect(task.deleteLater)
        task.start()

    def unlock(self):
        self.task = None
        for b in (self.file_button, self.camera_button, self.analyze_button):
            b.setEnabled(True)
        self.upload_button.setEnabled(self.folder is not None and bool(self.cfg["upload"]["url"]))

    def analyze(self):
        if self.frame is None:
            self.info.setText("请先采集或打开图片")
            return
        self.timer.stop()
        self.folder = None
        self.info.setText("分析中")
        self.start_task(Task(self.cfg, image=self.frame.copy(), parent=self))

    def upload(self):
        if self.folder is not None:
            self.info.setText("上传中")
            self.start_task(Task(self.cfg, folder=self.folder, parent=self))

    def finished_result(self, data):
        if "http" in data:
            self.info.setText(f"上传完成：HTTP {data['http']}")
            return
        result = data["result"]
        self.show_image(data["image"])
        self.folder = data["folder"] if result["needs_review"] else None
        summary = "；".join(f"{x['label']} {x['classification_confidence']:.3f}" for x in result["cells"])
        self.info.setText(f"检测到 {len(result['cells'])} 个候选细胞。{summary}\n"
                          f"待复核：{result['needs_review']}。保存目录：{data['folder']}")

    def closeEvent(self, event):
        if self.task is not None and self.task.isRunning():
            self.info.setText("当前操作正在执行，结束后可关闭窗口")
            event.ignore()
            return
        self.stop_camera()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/edge.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    app = QApplication(sys.argv[:1])
    window = Window(cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
