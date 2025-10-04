import numpy as np
import argparse
import cv2 as cv
import subprocess
import datetime
from yolo_utils import infer_image, show_image

FLAGS = []
green = cv.imread('yellow.jpeg')
red = cv.imread('redSignal.jpeg')
ambulanceDetected = False
isGreen = False
turn_green = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', type=str, default='./model/', help='Path to model files.')
    parser.add_argument('-w', '--weights', type=str, default='./model/yolov3.weights', help='YOLOv3 weights file.')
    parser.add_argument('-cfg', '--config', type=str, default='./model/yolov3.cfg', help='YOLOv3 config file.')
    parser.add_argument('-i', '--image-path', type=str, help='Path to image file.')
    parser.add_argument('-v', '--video-path', type=str, help='Path to video file.')
    parser.add_argument('-vo', '--video-output-path', type=str, default='./output/output.avi', help='Path of output video.')
    parser.add_argument('-l', '--labels', type=str, default='./model/coco-labels', help='Labels file path.')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Min confidence.')
    parser.add_argument('-th', '--threshold', type=float, default=0.3, help='Non-max suppression threshold.')
    parser.add_argument('--download-model', type=bool, default=False, help='Download YOLOv3 model if missing.')
    parser.add_argument('-t', '--show-time', type=bool, default=False, help='Show inference time.')

    FLAGS, _ = parser.parse_known_args()

    if FLAGS.download_model:
        subprocess.call(['./model/get_model.sh'])

    labels = open(FLAGS.labels).read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    net = cv.dnn.readNetFromONNX('./model/yolov3.onnx')
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    if FLAGS.image_path:
        img = cv.imread(FLAGS.image_path)
        height, width = img.shape[:2]
        img, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
        show_image(img)

    elif FLAGS.video_path:
        vid = cv.VideoCapture(FLAGS.video_path)
        height, width = None, None
        writer = None

        while True:
            grabbed, frame = vid.read()
            if not grabbed:
                break
            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)
            if writer is None:
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            writer.write(frame)

            cv.imshow('Traffic Camera', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        writer.release()
        vid.release()
        cv.destroyAllWindows()
