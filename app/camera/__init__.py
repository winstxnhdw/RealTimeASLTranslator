from collections import deque
from threading import Thread

from cv2 import VideoCapture
from typing_extensions import Self

class Camera:

    def __init__(self, capture_id: int=0, buffer_size: int=1):
        
        self.capture_id = capture_id
        self.buffer_size = buffer_size
        self.capture: VideoCapture
        self.capture_thread: Thread
        self.buffer = deque()
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

            _, frame = self.capture.read()

            if len(self.buffer) == self.buffer_size:
                self.buffer.popleft()

            self.buffer.append(frame)


    def is_capturing(self) -> bool:

        print(f"Filling buffer: {len(self.buffer)}/{self.buffer_size}")
        return len(self.buffer) == self.buffer_size
            