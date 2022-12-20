from sys import argv
from time import sleep

import cv2 as cv

from app.camera import Camera
from app.translator import video_to_asl
from app.server import HTTPDaemon, PredictedResult

from app.translator import load_model, load_vocabulary
from app.config import Config

# from app.translator import video_to_asl

def camera_loop(camera: Camera, model, word_data):

    retry_count = 0

    while not camera.is_capturing():
        if retry_count > 3:
            raise Exception("No camera frames found.")

        retry_count += 1
        sleep(1)

    while True:
        cv.imshow('Input', camera.buffer[-1])
        PredictedResult.out = video_to_asl(video = camera.buffer, confidence = 0.7, model = model, word_data = word_data)
        print(PredictedResult.out)

        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()


def main():
    model = load_model(Config.checkpoint_path, Config.number_of_classes, Config.number_of_frames)
    word_data = load_vocabulary(Config.vocabulary_path)

    # sleep(10000)
    with Camera(0, 64) as camera:
        camera_loop(camera, model, word_data)


def init_server():

    port = parse_args()
    host = 'localhost'

    with HTTPDaemon(host, port):
        try:
            main()
            
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