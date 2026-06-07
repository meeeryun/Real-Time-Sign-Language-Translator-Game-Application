import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 설정 
DATASET_DIR  = Path("dataset_custom")       # 관련 라벨들 저장 파일 설정
MODEL_PATH   = "model_lstm.keras"           # 모델 생성 및 경로 설정
ENCODER_PATH = "label_encoder.pkl"          # 인코더 생성 및 경로 설정
SEQ_LEN      = 60                           # 시퀀스 30은 너무 짧아서 60으로 늘리기
FEATURE_DIM  = 126                          # 양손 (63 × 2) = 126차원
EPOCHS       = 100                          # 에포크 수
BATCH_SIZE   = 32                           # Batch 사이즈

# Data Load
X, y = [], []

for word_dir in sorted(DATASET_DIR.iterdir()):
    if not word_dir.is_dir():
        continue
    files = sorted(word_dir.glob("*.npy"))
    if not files:
        continue
    for npy_file in files:
        seq = np.load(npy_file)
        if seq.shape == (SEQ_LEN, FEATURE_DIM):
            X.append(seq)
            y.append(word_dir.name)
        else:
            print(f"스킵 (shape 불일치): {npy_file.name} → {seq.shape}")

if not X:
    print("dataset_words에 올바른 데이터가 없습니다.")
    print("collect_word_data.py를 먼저 실행해서 데이터를 수집하세요.")
    exit()

X = np.array(X, dtype=np.float32)
print(f"총 {len(X)}개 시퀀스 로드")
print(f"클래스 ({len(set(y))}개): {sorted(set(y))}\n")

# 인코더 생성
le          = LabelEncoder()
y_enc       = le.fit_transform(y)
num_classes = len(le.classes_)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"라벨 인코더 저장: {ENCODER_PATH}")

# train, test 7:3 분할 (7:3 이유: 8:2로 했을 때 시퀀스가 2개인 친구들의 오류로 인해서 7:3으로 변경)
use_stratify = num_classes > 1 and min(np.bincount(y_enc)) >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.3, # 7:3
    random_state=42,
    stratify=y_enc if use_stratify else None,
)

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes)

# 모델 설정
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.4),
    layers.LSTM(64),
    layers.Dropout(0.4),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
], name="ksl_word_lstm")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# Train
print("\n학습 시작...\n")

cb_list = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
]

model.fit(
    X_train, y_train_oh,
    validation_data=(X_test, y_test_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
)

# TEST & TRAIN 평가
loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n테스트 정확도 : {acc * 100:.1f}%")
print(f"모델 저장     : {MODEL_PATH}")
print(f"인코더 저장   : {ENCODER_PATH}")