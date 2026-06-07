# DPI 오류 메시지를 없애기 위함
import sys
import os

#------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#-----------------------------------------------------
# Tensorflow 쓸데없는 오류 메시지 가리기

from PyQt6.QtWidgets import QApplication
from ui.menu_page import MenuPage

def main():
    app = QApplication(sys.argv)
    window = MenuPage()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
