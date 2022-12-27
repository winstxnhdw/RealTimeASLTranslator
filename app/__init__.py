from sys import argv
from time import sleep

import cv2 as cv

from app.camera import Camera
from app.server import HTTPDaemon
from app.translator import Translator


def camera_loop(camera: Camera, translator: Translator):

    retry_count = 0

    while not camera.is_capturing():
        if retry_count > 3:
            raise Exception("No camera frames found.")

        retry_count += 1
        sleep(1)

    while True:
        cv.imshow('Input', camera.buffer[-1])
        translator.video_to_asl(camera.buffer)

        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()


def main(translator: Translator):

    with Camera(0, 64) as camera:
        camera_loop(camera, translator)


def init_server():

    host = 'localhost'
    port = parse_args()
    translator = Translator(confidence=0.7)

    with HTTPDaemon(host, port, translator):
        try:
            main(translator)
            
        except KeyboardInterrupt:
            print("\nManual exit detected.")

        finally:
            print("Exiting..")


def parse_args() -> int:

    try:
        return 5000 if len(argv) < 2 else int(argv[1])

    except ValueError:
        print("\nPort must be an integer.\ne.g. python server.py 5000\n")
        raise