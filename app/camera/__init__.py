from threading import Thread

from cv2 import VideoCapture
from numpy import ndarray
from typing_extensions import Self


class Camera:

    def __init__(self, capture_id: int=0):
        
        self.capture_id = capture_id
        self.capture: VideoCapture
        self.capture_thread: Thread
        self.buffer: ndarray
        self.stop_capture = False


    def __enter__(self) -> Self:

        self.capture = VideoCapture(self.capture_id)

        if not self.capture.isOpened():
            raise IOError("Unable to open device.")

        self.capture_thread = Thread(target=self.start_capture)
        self.capture_thread.start()
        
        return self


    def __exit__(self, *_):

        self.stop_capture = True
        self.capture_thread.join()
        self.capture.release()


    def start_capture(self):

        while True:
            if self.stop_capture:
                break

            _, self.buffer = self.capture.read()


    def is_capturing(self) -> bool:

        return len(self.buffer) != 0
            