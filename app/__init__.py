from time import sleep

import cv2 as cv

from app.camera import Camera
from app.translator import video_to_asl

# example code
def camera_loop(camera: Camera):

    retry_count = 0

    while not camera.is_capturing():
        if retry_count > 3:
            raise Exception("No camera frames found.")

        retry_count += 1
        sleep(1)

    while True:
        cv.imshow('Input', camera.buffer[-1])
        print(video_to_asl(camera.buffer))

        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()


def main():

    with Camera(0, 64) as camera:
        camera_loop(camera)