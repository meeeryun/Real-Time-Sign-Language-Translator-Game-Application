import time
import collections
import cv2 as cv
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf


class WordRecognizer:
    def __init__(
        self,
        model_path="model_lstm.keras",
        encoder_path="label_encoder.pkl",
        margin_min=0.25,
        smooth_frames=3,
        use_motion_filter=True,
        cooldown=0.8,
    ):
        self.margin_min        = margin_min
        self.use_motion_filter = use_motion_filter
        self.cooldown          = cooldown
        self._last_pred_time   = 0.0

        self.model = tf.keras.models.load_model(model_path)
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,          
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.sequence      = collections.deque(maxlen=60)
        self.wrist_history = collections.deque(maxlen=30)
        self.prob_history  = collections.deque(maxlen=smooth_frames)
        self.frame_count   = 0

    def predict(self, frame) -> dict | None:
        rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        right_hand_data = [0.0] * 63
        left_hand_data = [0.0] * 63
        wrist_pos = None

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                
                wrist = hand_landmarks.landmark[0]
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                if hand_label == "Right":
                    right_hand_data = coords
                    wrist_pos = [wrist.x, wrist.y]
                elif hand_label == "Left":
                    left_hand_data = coords
                    if wrist_pos is None: 
                        wrist_pos = [wrist.x, wrist.y]

        if wrist_pos is None:
            self.sequence.clear()
            self.wrist_history.clear()
            self.prob_history.clear()
            self.frame_count = 0
            return None

        self.wrist_history.append(wrist_pos)
        combined_features = right_hand_data + left_hand_data
        self.sequence.append(combined_features)

        if len(self.sequence) < 60:
            return None

        if self.use_motion_filter and len(self.wrist_history) == 30:
            std_x = np.std([p[0] for p in self.wrist_history])
            std_y = np.std([p[1] for p in self.wrist_history])
            if std_x < 0.004 and std_y < 0.004:
                return None

        self.frame_count = (self.frame_count + 1) % 5
        if self.frame_count != 0:
            return None

        if time.time() - self._last_pred_time < self.cooldown:
            return None

        raw_probs = self.model.predict(
            np.expand_dims(list(self.sequence), axis=0), verbose=0
        )[0]

        self.prob_history.append(raw_probs)
        smoothed = np.mean(self.prob_history, axis=0)

        top2   = np.argsort(smoothed)[-2:]
        
        if len(smoothed) == 1:
            if smoothed[0] < self.margin_min:
                return None
            word = self.label_encoder.inverse_transform([0])[0]
        else:
            margin = smoothed[top2[-1]] - smoothed[top2[-2]]
            if margin < self.margin_min:
                return None
            word = self.label_encoder.inverse_transform([top2[-1]])[0]

        for _ in range(15):
            self.sequence.popleft()
        self.prob_history.clear()
        self._last_pred_time = time.time()

        return {"word": word, "confidence": float(smoothed[top2[-1]])}

    def release(self):
        self.detector.close()