import queue
import threading
import pyttsx3

class TTSManager:
    def __init__(self, rate=150):
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._worker, args=(rate,), daemon=True)
        self._thread.start()

    def _worker(self, rate): # pyttsx3 충돌 방지
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass # MAC/LINUX 는 pythoncom 필요 X

        while True:
            text = self._queue.get()
            if text is None:
                break

            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()
            
            del engine 

    def speak(self, text: str):
        if text:
            self._queue.put(text)

    def stop(self):
        self._queue.put(None)