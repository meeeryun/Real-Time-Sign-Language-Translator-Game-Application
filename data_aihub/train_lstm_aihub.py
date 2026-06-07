import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # Warning이 뜬다면 오류가 아닌 인식이 느린 이유 -> 실제로는 인식 되고 있음

# 설정
DATASET_DIR  = Path("dataset_words")
MODEL_PATH   = "model_lstm.keras"
ENCODER_PATH = "label_encoder.pkl"
SEQ_LEN      = 30   # 여기는 시퀀스 30 (어차피 사용자 인식이 아닌 본 위치벡터를 이용한 변환)
FEATURE_DIM  = 126  # 양손 63 + 63 = 126차원
EPOCHS       = 100
BATCH_SIZE   = 32

# DATA LOAD
X, y = [], []

for word_dir in sorted(DATASET_DIR.iterdir()):
    if not word_dir.is_dir():
        continue
    for npy_file in sorted(word_dir.glob("*.npy")):
        seq = np.load(npy_file)
        if seq.shape == (SEQ_LEN, FEATURE_DIM):
            X.append(seq)
            y.append(word_dir.name)

if not X:
    print("dataset_words에 조건에 맞는 데이터가 없습니다. 데이터의 차원(Shape)을 확인하세요.")
    exit()

X = np.array(X, dtype=np.float32)   # (N, 30, 126)
print(f"총 {len(X)}개 시퀀스 로드")
print(f"클래스 ({len(set(y))}개): {sorted(set(y))}\n")

# 인코더 생성
le        = LabelEncoder()
y_enc     = le.fit_transform(y)
num_classes = len(le.classes_)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"라벨 인코더 저장: {ENCODER_PATH}")

# train, test 8:2 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes)

print("X shape :", X.shape)
print("클래스 수 :", num_classes)

unique, counts = np.unique(y_enc, return_counts=True)

print("최소 샘플 수 :", counts.min())
print("최대 샘플 수 :", counts.max())
print("평균 샘플 수 :", counts.mean())

# MODEL
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.4),
    layers.LSTM(64),
    layers.Dropout(0.4),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
], name="ksl_lstm")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# TEST & TRAIN
cb_list = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
]

print("\n학습 시작...\n")
model.fit(
    X_train, y_train_oh,
    validation_data=(X_test, y_test_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
)

# TEST & TRAIN 평가 결과
loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n테스트 정확도 : {acc * 100:.1f}%")
print(f"모델 저장     : {MODEL_PATH}")
print(f"인코더 저장   : {ENCODER_PATH}")