import time
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFontDatabase
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout,
)

from core.recognition_manager import RecognitionManager
from core.tts_manager import TTSManager


class TranslatorPage(QWidget):

    def __init__(self, parent=None, go_back=None, is_dark_mode=False):
        super().__init__(parent)

        self.go_back      = go_back
        self.is_dark_mode = is_dark_mode

        self.setWindowTitle("KSL Translator")
        self.setObjectName("TranslatorPage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        QFontDatabase.addApplicationFont("fonts/Paperlogy-6SemiBold.ttf")
        QFontDatabase.addApplicationFont("fonts/Paperlogy-2ExtraLight.ttf")

        self.cap                  = cv.VideoCapture(0)
        self.recognition_manager  = RecognitionManager()
        self.tts                  = TTSManager()

        # 상태
        self.result_text    = ""
        self.word_overlay   = ""
        self.overlay_time   = 0
        self.last_word      = ""
        self.last_word_time = 0

        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # UI

    def setup_ui(self):
        root = QHBoxLayout(self)

        # 왼쪽: 카메라
        left = QVBoxLayout()
        left.addStretch()

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #222; border-radius: 16px;")
        left.addWidget(self.camera_label)
        root.addLayout(left)

        # 오른쪽: 정보 패널
        right = QVBoxLayout()

        title = QLabel("수화 번역")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("TranslatorTitle")
        title.setStyleSheet("font-family: Paperlogy; font-size: 32px; font-weight: 600;")
        right.addWidget(title)

        self.current_label = QLabel("-")
        self.current_label.setObjectName("CurrentTitle")
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_label.setStyleSheet(
            "font-family: Paperlogy; font-size: 60px; font-weight: bold;"
        )
        right.addWidget(self.current_label)

        self.conf_label = QLabel("")
        self.conf_label.setObjectName("ConfLabel")
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.conf_label.setStyleSheet("font-family: Paperlogy; font-size: 14px;")
        right.addWidget(self.conf_label)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setObjectName("ResultTextBox")
        self.text_box.setStyleSheet("font-size: 28px; border-radius: 12px;")
        right.addWidget(self.text_box)

        self.clear_btn = QPushButton("초기화")
        self.back_btn  = QPushButton("뒤로가기")

        for btn in [self.clear_btn, self.back_btn]:
            btn.setStyleSheet("font-family: Paperlogy; font-size: 15px; font-weight: 200;")

        self.clear_btn.clicked.connect(self.reset)
        if self.go_back:
            self.back_btn.clicked.connect(self.go_back)

        right.addWidget(self.clear_btn)
        right.addWidget(self.back_btn)

        self.apply_theme()
        root.addLayout(right)

    def apply_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet("""
                QWidget#TranslatorPage  { background-color: #1A1A24; }
                QLabel#TranslatorTitle  { font-family: Paperlogy; font-size: 30px; color: #FFFFFF; }
                QLabel#CurrentTitle     { color: #64D2FF; }
                QLabel#ConfLabel        { color: #888888; }
                QTextEdit#ResultTextBox { background-color: #262636; color: #FFFFFF;
                                          border: 1px solid #3A3A4C; border-radius: 12px; }
                QPushButton             { font-family: Paperlogy; background-color: #2D2D3A;
                                          color: #FFFFFF; border-radius: 10px; padding: 10px; }
                QPushButton:hover       { background-color: #3D3D4A; }
            """)
        else:
            self.setStyleSheet("""
                QWidget#TranslatorPage  { background-color: #F5F5F7; }
                QLabel#TranslatorTitle  { font-family: Paperlogy; font-size: 30px; color: #1D1D1F; }
                QLabel#CurrentTitle     { color: #0A84FF; }
                QLabel#ConfLabel        { color: #888888; }
                QTextEdit#ResultTextBox { background-color: #FFFFFF; color: #1D1D1F;
                                          border: 1px solid #D2D2D7; border-radius: 12px; }
                QPushButton             { font-family: Paperlogy; background-color: #FFFFFF;
                                          color: #1D1D1F; border: 1px solid #D2D2D7;
                                          border-radius: 10px; padding: 10px; }
                QPushButton:hover       { background-color: #E8E8ED; }
            """)

    # MAIN

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv.resize(frame, (800, 600))
        frame = cv.flip(frame, 1)

        result = self.recognition_manager.process(frame)

        if result and result["type"] == "word":
            word         = result["text"]
            conf         = result["confidence"]
            current_time = time.time()

            self.current_label.setText(word)
            self.conf_label.setText(f"신뢰도: {int(conf * 100)}%")

            self.word_overlay = word
            self.overlay_time = current_time

            # 중복 방지: 같은 단어는 2초 이내 재추가 안 함
            if word != self.last_word or (current_time - self.last_word_time > 2.0):
                if self.result_text and not self.result_text.endswith(" "):
                    self.result_text += " "
                self.result_text += word + " "
                self.text_box.setText(self.result_text)

                self.last_word      = word
                self.last_word_time = current_time

                self.tts.speak(word)  # TTS

        # CAMERA OVERLAY
        if self.word_overlay and (time.time() - self.overlay_time < 2.0):
            try:
                pil_img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                draw    = ImageDraw.Draw(pil_img)
                font    = ImageFont.truetype("fonts/Paperlogy-6SemiBold.ttf", 60)
                draw.rectangle([(120, 450), (680, 540)], fill=(0, 0, 0))
                draw.text((160, 460), self.word_overlay, font=font, fill=(255, 255, 255))
                frame   = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
            except Exception:
                pass
        elif time.time() - self.overlay_time >= 2.0:
            self.word_overlay = ""

        rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    # Initialization / Exit

    def reset(self):
        self.result_text    = ""
        self.last_word_time = 0
        self.word_overlay   = ""
        self.current_label.setText("-")
        self.conf_label.setText("")
        self.text_box.clear()
        self.recognition_manager.reset()

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.recognition_manager.release()
        self.tts.stop()
        event.accept()