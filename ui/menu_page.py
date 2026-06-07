import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase

from ui.translator_page import TranslatorPage
from ui.game_page import GamePage


class MenuPage(QWidget):
    def __init__(self):
        super().__init__()

        QFontDatabase.addApplicationFont("fonts/GmarketSansTTFBold.ttf")
        QFontDatabase.addApplicationFont("fonts/GmarketSansTTFMedium.ttf")
        QFontDatabase.addApplicationFont("fonts/Paperlogy-6SemiBold.ttf")

        self.setWindowTitle("KSL Learning")
        self.resize(1200, 800)
        self.setObjectName("MenuPage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.is_dark_mode = False

        layout = QVBoxLayout()

        # 다크모드 버튼 (우상단)
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.theme_btn = QPushButton("다크 모드")
        self.theme_btn.setFixedSize(120, 40)
        self.theme_btn.setObjectName("theme_btn")
        self.theme_btn.clicked.connect(self.toggle_theme)
        top_layout.addWidget(self.theme_btn)

        # 타이틀
        self.title = QLabel("KSL Learning")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setObjectName("MainTitle")
        self.title.setStyleSheet(
            'font-family: "GmarketSansTTF"; font-size: 40px; font-weight: bold;'
        )

        # 메뉴 버튼
        translator_btn = QPushButton("수화 번역")
        translator_btn.setFixedSize(350, 150)
        translator_btn.setStyleSheet(
            'font-family: "Paperlogy"; font-size: 24px; font-weight: 600;'
        )

        game_btn = QPushButton("수화 게임")
        game_btn.setFixedSize(350, 150)
        game_btn.setStyleSheet(
            'font-family: "Paperlogy"; font-size: 24px; font-weight: 600;'
        )

        translator_btn.clicked.connect(self.open_translator)
        game_btn.clicked.connect(self.open_game)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(translator_btn)
        button_layout.addSpacing(80)
        button_layout.addWidget(game_btn)
        button_layout.addStretch()

        layout.addLayout(top_layout)
        layout.addStretch()
        layout.addWidget(self.title)
        layout.addSpacing(60)
        layout.addLayout(button_layout)
        layout.addStretch()

        self.setLayout(layout)
        self.apply_theme()

    # DARK/LIGHT 모드 테마 적용

    def apply_theme(self):
        if self.is_dark_mode:
            self.theme_btn.setText("라이트 모드")
            self.setStyleSheet("""
                QWidget#MenuPage  { background-color: #1A1A24; }
                QLabel#MainTitle  { font-family: "GmarketSansTTF"; font-size: 45px;
                                    font-weight: bold; color: #FFFFFF; }
                QPushButton       { font-family: "GmarketSansTTF"; font-size: 24px;
                                    background-color: #2D2D3A; color: #FFFFFF;
                                    border: none; border-radius: 15px; }
                QPushButton:hover { background-color: #3D3D4A; }
                QPushButton#theme_btn { font-size: 14px; background-color: #333344; }
            """)
        else:
            self.theme_btn.setText("다크 모드")
            self.setStyleSheet("""
                QWidget#MenuPage  { background-color: #F5F5F7; }
                QLabel#MainTitle  { font-family: "GmarketSansTTF"; font-size: 45px;
                                    font-weight: bold; color: #1D1D1F; }
                QPushButton       { font-family: "GmarketSansTTF"; font-size: 24px;
                                    background-color: #FFFFFF; color: #1D1D1F;
                                    border: 1px solid #D2D2D7; border-radius: 15px; }
                QPushButton:hover { background-color: #E8E8ED; }
                QPushButton#theme_btn { font-size: 14px; background-color: #E8E8ED; }
            """)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    # Page 전환

    def open_translator(self):  # 수화 번역 버튼 이벤트 -> 수화 번역 창
        self.translator = TranslatorPage(
            go_back=self.reopen_from_translator,
            is_dark_mode=self.is_dark_mode,
        )
        self.translator.show()
        self.hide()

    def reopen_from_translator(self): # 창을 한번 켰다 끄면 스택이 없어져서 다시 못 여는 오류 방지를 위해 reopen 함수 생성
        self.translator.close()
        self.show()

    def open_game(self):    # 수화 게임 버튼 이벤트 -> 수화 게임 창
        self.game = GamePage(
            go_back=self.reopen_from_game,
            is_dark_mode=self.is_dark_mode,
        )
        self.game.show()
        self.hide()

    def reopen_from_game(self):     # 위와 동일
        self.game.close()
        self.show()