from core.word_recognizer import WordRecognizer


class RecognitionManager:
    def __init__(
        self,
        word_model_path="model_lstm.keras",
        encoder_path="label_encoder.pkl",
    ):
        self.word_recognizer = WordRecognizer(
            model_path=word_model_path,
            encoder_path=encoder_path,
        )

    def process(self, frame):
        result = self.word_recognizer.predict(frame)
        if result:
            return {
                "type":       "word",
                "text":       result["word"],
                "confidence": result["confidence"],
            }
        return None

    def release(self):
        self.word_recognizer.release()

    def reset(self):
        self.word_recognizer.sequence.clear()
        self.word_recognizer.wrist_history.clear()
        self.word_recognizer.prob_history.clear()