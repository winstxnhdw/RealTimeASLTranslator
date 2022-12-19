from time import sleep

import cv2 as cv

from app.camera import Camera

# example code
def camera_loop(camera: Camera):

    retry_count = 0

    while not camera.is_capturing():
        if retry_count > 3:
            raise Exception("No camera frames found.")

        retry_count += 1
        sleep(1)

    while True:
        cv.imshow('Input', camera.buffer)

        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()


def main():
    
    cv.namedWindow('Camera', cv.WINDOW_GUI_NORMAL)

    with Camera(0) as camera:
        camera_loop(camera)