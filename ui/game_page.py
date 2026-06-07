import os
import random
import time
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFontDatabase
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QDialog
)

from core.recognition_manager import RecognitionManager
from core.tts_manager import TTSManager

# 게임 STATE
IDLE      = "idle"
PLAYING   = "playing"
FEEDBACK  = "feedback"    # O / X 표시 중 (1초)
GAME_OVER = "game_over"   # X 시에

FONT_PATH = "fonts/Paperlogy-6SemiBold.ttf"


# POPUP (GAME OVER DIALOG)
class GameOverDialog(QDialog):
    def __init__(self, score, parent=None, is_dark_mode=False):
        super().__init__(parent)
        self.setWindowTitle("게임 종료")
        self.setFixedSize(320, 220)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title_label = QLabel("GAME OVER")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-family: Paperlogy; font-size: 26px; font-weight: bold; color: #FF3B30;")
        layout.addWidget(title_label)

        score_label = QLabel(f"점수: {score}개")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_label.setStyleSheet("font-family: Paperlogy; font-size: 20px; font-weight: 600;")
        layout.addWidget(score_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.retry_btn = QPushButton("다시하기")
        self.lobby_btn = QPushButton("로비로")
        
        for btn in [self.retry_btn, self.lobby_btn]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("font-family: Paperlogy; font-size: 14px; font-weight: 500;")
            btn_layout.addWidget(btn)
            
        layout.addLayout(btn_layout)

        # 버튼 이벤트 연결
        self.retry_btn.clicked.connect(self.accept)  # QDialog.DialogCode.Accepted 반환
        self.lobby_btn.clicked.connect(self.reject)  # QDialog.DialogCode.Rejected 반환

        # DARK/WHITE 모드 테마 적용
        if is_dark_mode:
            self.setStyleSheet("""
                QDialog { background-color: #1A1A24; }
                QLabel { color: #FFFFFF; }
                QPushButton { background-color: #2D2D3A; color: #FFFFFF; border-radius: 8px; }
                QPushButton:hover { background-color: #3D3D4A; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background-color: #F5F5F7; }
                QLabel { color: #1D1D1F; }
                QPushButton { background-color: #FFFFFF; color: #1D1D1F; border: 1px solid #D2D2D7; border-radius: 8px; }
                QPushButton:hover { background-color: #E8E8ED; }
            """)


# GAME PAGE
class GamePage(QWidget):

    def __init__(self, parent=None, go_back=None, is_dark_mode=False):
        super().__init__(parent)

        self.go_back      = go_back
        self.is_dark_mode = is_dark_mode

        self.setWindowTitle("KSL Game")
        self.setObjectName("GamePage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        QFontDatabase.addApplicationFont("fonts/Paperlogy-6SemiBold.ttf")
        QFontDatabase.addApplicationFont("fonts/Paperlogy-2ExtraLight.ttf")

        self.cap                 = cv.VideoCapture(0)
        self.recognition_manager = RecognitionManager()
        self.tts                 = TTSManager()

        self.word_list           = []
        self.state               = IDLE
        self.correct_count       = 0
        self.target_word         = ""
        self.current_recognized  = ""   # 오버레이에 표시되는 현재 인식 단어
        self.feedback_text       = ""   # "O" or "X"
        self.feedback_correct    = False
        self.feedback_time       = 0.0

        self.setup_ui()
        self.apply_theme()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # UI

    def setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # 왼쪽: 카메라 + 문제 단어
        left = QVBoxLayout()
        left.setSpacing(12)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 500)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #222; border-radius: 16px;")
        left.addWidget(self.camera_label)

        self.word_label = QLabel("게임 시작을 눌러주세요")
        self.word_label.setObjectName("WordLabel")
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setFixedHeight(100)
        self.word_label.setStyleSheet(
            "font-family: Paperlogy; font-size: 48px; font-weight: bold; border-radius: 12px;"
        )
        left.addWidget(self.word_label)

        root.addLayout(left)

        # 오른쪽: Score 카운트 + 버튼
        right_widget = QWidget()    
        right_widget.setFixedWidth(260)

        right = QVBoxLayout(right_widget)    
        right.setSpacing(12)
        right.setContentsMargins(0, 0, 0, 0)

        title = QLabel("수화 게임")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("GameTitle")
        title.setStyleSheet("font-family: Paperlogy; font-size: 28px; font-weight: 600;")
        right.addWidget(title)

        right.addStretch()

        count_hint = QLabel("맞춘 개수")
        count_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_hint.setStyleSheet("font-family: Paperlogy; font-size: 14px; color: gray;")
        right.addWidget(count_hint)

        self.count_label = QLabel("0")
        self.count_label.setObjectName("ScoreLabel")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setStyleSheet(
            "font-family: Paperlogy; font-size: 80px; font-weight: bold;"
        )
        right.addWidget(self.count_label)

        self.result_label = QLabel("0")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-family: Paperlogy; font-size: 14px;")
        right.addWidget(self.result_label)

        right.addStretch()

        self.start_btn = QPushButton("게임 시작")
        self.start_btn.setMinimumHeight(60)
        self.start_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_btn.clicked.connect(self.start_game)

        self.back_btn = QPushButton("뒤로가기")
        self.back_btn.setMinimumHeight(50)
        self.back_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if self.go_back:
            self.back_btn.clicked.connect(self.go_back)

        for btn in [self.start_btn, self.back_btn]:
            btn.setStyleSheet("font-family: Paperlogy; font-size: 15px; padding: 8px;")
            right.addWidget(btn)

        root.addWidget(right_widget)

    def apply_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet("""
                QWidget#GamePage   { background-color: #1A1A24; }
                QLabel#GameTitle   { color: #FFFFFF; }
                QLabel#WordLabel   { background-color: #262636; color: #FFFFFF;
                                     border: 1px solid #3A3A4C; }
                QLabel#ScoreLabel  { color: #64D2FF; }
                QLabel#ResultLabel { color: #FF453A; }
                QPushButton        { font-family: Paperlogy; background-color: #2D2D3A;
                                     color: #FFFFFF; border-radius: 10px; padding: 10px; }
                QPushButton:hover  { background-color: #3D3D4A; }
            """)
        else:
            self.setStyleSheet("""
                QWidget#GamePage   { background-color: #F5F5F7; }
                QLabel#GameTitle   { color: #1D1D1F; }
                QLabel#WordLabel   { background-color: #FFFFFF; color: #1D1D1F;
                                     border: 1px solid #D2D2D7; }
                QLabel#ScoreLabel  { color: #0A84FF; }
                QLabel#ResultLabel { color: #FF3B30; }
                QPushButton        { font-family: Paperlogy; background-color: #FFFFFF;
                                     color: #1D1D1F; border: 1px solid #D2D2D7;
                                     border-radius: 10px; padding: 10px; }
                QPushButton:hover  { background-color: #E8E8ED; }
            """)

    # 키 입력 이벤트

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.state == PLAYING:
            self.confirm_answer()
        super().keyPressEvent(event)

    # GMAE LOGIC 

    def load_words_from_directory(self): # dataset_custom에서 문제 단어 가져와서 리스트업
        path = "dataset_custom"
        if os.path.exists(path) and os.path.isdir(path):
            words = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            return words
        return []

    def start_game(self):
        self.word_list = self.load_words_from_directory()

        if not self.word_list:
            self.word_label.setText("dataset_custom 폴더가 비어있습니다")
            return

        self.correct_count = 0
        self.count_label.setText("0")
        self.result_label.setText("Space: 인식된 단어 확정")
        self.state = PLAYING
        self.start_btn.setText("진행 중...")
        self.start_btn.setEnabled(False)
        self.next_word()

    def next_word(self):
        if not self.word_list:
            return
        self.target_word        = random.choice(self.word_list)
        self.current_recognized = ""
        self.word_label.setText(self.target_word)
        self.recognition_manager.reset()

    def confirm_answer(self): # 답 확정 (Space bar)
        if not self.current_recognized:
            return  # 아직 인식된 단어 없으면 무시

        if self.current_recognized == self.target_word:
            self.correct_count += 1
            self.count_label.setText(str(self.correct_count))
            self.feedback_text    = "O"
            self.feedback_correct = True
            self.tts.speak("정답")
        else:
            self.feedback_text    = "X"
            self.feedback_correct = False
            self.tts.speak("오답")

        self.feedback_time      = time.time()
        self.current_recognized = ""
        self.state              = FEEDBACK

    def show_game_over(self):
        self.state = GAME_OVER
        self.word_label.setText("게임 종료")
        self.recognition_manager.reset()

        self.timer.stop()

        # Game over 팝업 호출
        dialog = GameOverDialog(self.correct_count, self, self.is_dark_mode)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # '다시하기'를 누른 경우 재시작
            self.timer.start(30)
            self.start_game()
        else:
            # '로비로'를 누르거나 창을 닫은 경우 메인메뉴(menu_page.py)로 이동
            self.start_btn.setText("게임 시작")
            self.start_btn.setEnabled(True)
            self.result_label.setText("게임이 종료되었습니다.")
            if self.go_back:
                self.go_back()

    # Frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv.resize(frame, (800, 500))
        frame = cv.flip(frame, 1)
        h, w  = frame.shape[:2]

        # 카메라 인식
        if self.state == PLAYING:
            result = self.recognition_manager.process(frame)
            if result and result["type"] == "word":
                self.current_recognized = result["text"]

        # O or X 판단
        elif self.state == FEEDBACK:
            if time.time() - self.feedback_time > 1.0:
                if self.feedback_correct:
                    self.next_word()
                    self.state = PLAYING
                else:
                    # 1초간 X가 표시된 후 여기에 진입해 팝업을 띄웁니다.
                    self.show_game_over()
                self.feedback_text = ""

        # Camera Overlay
        frame = self._draw_overlay(frame, w, h)

        rgb      = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        qt_img   = QImage(rgb.data, w2, h2, ch * w2, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    def _draw_overlay(self, frame, w, h) -> np.ndarray:
        try:
            # O / X 피드백
            if self.state == FEEDBACK and self.feedback_text:
                pil_img    = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                draw       = ImageDraw.Draw(pil_img)
                font_large = ImageFont.truetype(FONT_PATH, 200)
                color      = (0, 200, 80) if self.feedback_correct else (220, 50, 50)
                bbox       = draw.textbbox((0, 0), self.feedback_text, font=font_large)
                tx         = (w - (bbox[2] - bbox[0])) // 2
                ty         = (h - (bbox[3] - bbox[1])) // 2 - 20
                draw.text((tx, ty), self.feedback_text, font=font_large, fill=color)
                return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

            # 인식된 단어 오버레이 (하단)
            elif self.state == PLAYING and self.current_recognized:
                overlay = frame.copy()
                cv.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
                frame = cv.addWeighted(overlay, 0.65, frame, 0.35, 0)

                pil_img  = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                draw     = ImageDraw.Draw(pil_img)
                font_mid = ImageFont.truetype(FONT_PATH, 52)
                font_sm  = ImageFont.truetype(FONT_PATH, 22)
                draw.text((20, h - 82), self.current_recognized,
                          font=font_mid, fill=(255, 255, 255))
                draw.text((20, h - 28), "Space: 확정",
                          font=font_sm, fill=(180, 180, 180))
                return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        except Exception:
            pass

        return frame

    # EXIT

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.recognition_manager.release()
        self.tts.stop()
        event.accept()