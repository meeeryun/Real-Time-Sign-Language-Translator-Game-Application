import os
import time
import numpy as np
import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# 설정
DATASET_DIR = "dataset_custom"              # 단어 저장할 파일 설정
SEQ_LEN     = 60                            # 시퀀스 60 통일 (convert_data_mine.py, word_recognizer.py, train_data_mine.py)
os.makedirs(DATASET_DIR, exist_ok=True) 

font_path = "fonts/Paperlogy-6SemiBold.ttf" # 폰트 설정

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 양손
    min_detection_confidence=0.7, # 최소 신뢰도 70% 설정
    min_tracking_confidence=0.7, 
)

# UI
def put_kor(frame, text, pos, color=(255, 255, 255), size=26):
    try:
        font    = ImageFont.truetype(font_path, size)
        img     = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame


def extract_features(result) -> list:   # 손 랜드마크 잡기
    right = [0.0] * 63
    left  = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for lm, hand in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand.classification[0].label
            wrist = lm.landmark[0]
            coords = []
            for p in lm.landmark:
                coords.extend([p.x - wrist.x, p.y - wrist.y, p.z - wrist.z])
            if label == "Right":
                right = coords
            else:
                left = coords

    return right + left  # 63 + 63 = 126 


def get_seq_count(word: str) -> int:    # dataset_custom에 시퀀스 파일들 넣기
    d = os.path.join(DATASET_DIR, word)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


# MAIN 함수: 데이터 수집 + UI + 데이터 저장
def main():
    word = input("수집할 단어를 입력하세요: ").strip()
    if not word:
        print("단어가 비어있습니다.")
        return

    out_dir = os.path.join(DATASET_DIR, word)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("데이터 수집", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("데이터 수집", 960, 720)

    state           = "idle"       # idle / countdown / collecting
    countdown_start = 0.0
    sequence        = []
    seq_idx         = get_seq_count(word)

    print(f"\n[{word}] 현재 {seq_idx}개 시퀀스 수집됨")
    print("스페이스바: 수집 시작  |  ESC/q: 종료\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        hand_detected = result.multi_hand_landmarks is not None
        if hand_detected:
            for lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # STATE
        if state == "countdown":
            remaining = 3 - int(time.time() - countdown_start)
            if remaining > 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
                frame = put_kor(frame, str(remaining),
                                (w // 2 - 20, h // 2 - 60),
                                color=(0, 255, 255), size=120)
            else:
                state    = "collecting"
                sequence = []

        elif state == "collecting":
            if hand_detected:
                sequence.append(extract_features(result))

            bar_w = int((len(sequence) / SEQ_LEN) * (w - 40))
            cv2.rectangle(frame, (0, h - 32), (w, h), (30, 30, 30), -1)
            cv2.rectangle(frame, (20, h - 26), (20 + bar_w, h - 6), (0, 210, 90), -1)
            frame = put_kor(frame, f"수집 중: {len(sequence)}/{SEQ_LEN}",
                            (10, h - 60), color=(0, 210, 90))

            if len(sequence) >= SEQ_LEN:
                arr  = np.array(sequence[:SEQ_LEN], dtype=np.float32)
                path = os.path.join(out_dir, f"seq_{seq_idx:04d}.npy")
                np.save(path, arr)
                seq_idx += 1
                print(f"  [{word}] seq_{seq_idx - 1:04d} 저장 완료 (총 {seq_idx}개)")
                state = "idle"

        # ── 상단 상태바 ───────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)
        hand_color = (0, 255, 100) if hand_detected else (0, 80, 255)
        hand_text  = "손 감지됨" if hand_detected else "손 없음"
        frame = put_kor(frame,
                        f"{word}  |  {seq_idx}개 수집됨  |  {hand_text}",
                        (10, 12), color=hand_color)

        if state == "idle":
            frame = put_kor(frame, "스페이스바: 수집 시작  |  ESC: 종료",
                            (10, h - 30), color=(180, 180, 180), size=20)

        cv2.imshow("데이터 수집", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" ") and state == "idle":
            state           = "countdown"
            countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n수집 완료: [{word}] 총 {seq_idx}개 시퀀스")


if __name__ == "__main__":
    main()