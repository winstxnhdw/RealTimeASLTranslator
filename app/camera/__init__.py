from threading import Thread

from cv2 import VideoCapture
from typing_extensions import Self


class Camera:

    def __init__(self, capture_id: int=0):
        
        self.capture_id = capture_id
        self.capture: VideoCapture
        self.capture_process: Thread
        self.stop_capture = False
        self.buffer = []


    def __enter__(self) -> Self:

        self.capture = VideoCapture(self.capture_id)

        if not self.capture.isOpened():
            raise IOError("Unable to open device.")

        self.capture_process = Thread(target=self.start_capture)
        self.capture_process.start()
        
        return self


    def __exit__(self, *_):

        if not self.capture_process:
            return

        self.stop_capture = True
        self.capture_process.join()
        self.capture.release()


    def start_capture(self):

        while True:
            if self.stop_capture:
                break

            _, self.buffer = self.capture.read()


    def is_capturing(self) -> bool:

        return len(self.buffer) != 0
            