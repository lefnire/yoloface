# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************


import argparse, sys, os, colorsys, cv2, pdb
import numpy as np
from models.yoloface.utils import *
from models.model import Model
from PIL import Image

from models.yoloface.model import eval

from keras import backend as K
from keras.models import load_model
from timeit import default_timer as timer
from PIL import ImageDraw, Image

class Yoloface_Model(Model):
    def __init__(self, args):
        args.model_cfg = 'models/yoloface/cfg/yolov3-face.cfg'
        super().__init__(args)


class Yoloface_GPU(Yoloface_Model):
    devices = ['GPU']

    def __init__(self, args):
        args.model_weights = 'models/yoloface/model-weights/YOLO_Face.h5'
        args.anchors = 'models/yoloface/cfg/yolo_anchors.txt'  # path to anchor definitions
        args.classes = 'models/yoloface/cfg/face_classes.txt'  # path to class definitions'
        args.score = 0.5  # the score threshold
        args.iou = 0.45  # the iou threshold
        args.img_size = (416, 416)  # input image size
        args.image = False  # image detection mode
        args.video = 'samples/subway.mp4'  # path to the video
        args.output = 'outputs/'  # image/video output path

        # FIXME
        self.args = args
        self.model_path = args.model_weights
        self.classes_path = args.classes
        self.anchors_path = args.anchors
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate()
        self.model_image_size = args.img_size

        super().__init__(args)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        # print(class_names)
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file'

        # Load model, or construct model and load weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # print('[i] ==> {} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(102)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names),
                                           self.input_image_shape,
                                           score_threshold=self.args.score,
                                           iou_threshold=self.args.iou)
        return boxes, scores, classes

    def letterbox_image(self, image, size):
        """Resize image with unchanged aspect ratio using padding"""
        img_width, img_height = image.size
        w, h = size
        scale = min(w / img_width, h / img_height)
        nw = int(img_width * scale)
        nh = int(img_height * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def detect_image(self, frame):
        image = Image.fromarray(frame.astype('uint8'), 'RGB')
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = self.letterbox_image(
                image,
                tuple(reversed(self.model_image_size))
            )
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        # Add batch dimension
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        thickness = (image.size[0] + image.size[1]) // 400

        found = False
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # text = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # print(text, (left, top), (right, bottom))
            found = True
            cv2.rectangle(frame, (left, top), (right, bottom), self.color, 2)

            # for thk in range(thickness):
            #     draw.rectangle(
            #         [left + thk, top + thk, right - thk, bottom - thk],
            #         outline=(51, 178, 255))
            # del draw

        return image, out_boxes, found

    def close_session(self):
        self.sess.close()

    def draw_boxes(self, frame):
        _, _, found = self.detect_image(frame)
        return found

    def close(self):
        self.close_session()

class Yoloface_CPU(Yoloface_Model):
    devices = ['CPU']

    def __init__(self, args):
        args.model_weights = 'models/yoloface/model-weights/yoloface.weights'
        super().__init__(args)
        # Give the configuration and weight files for the model and load the network
        # using them.
        net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net = net

    def draw_boxes(self, frame):
        net = self.net

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        boxes = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        # return: is this array of bounding-boxes NOT empty?
        return bool(boxes)

        # initialize the set of information we'll displaying on the frame
        # info = [('number of faces detected', '{}'.format(len(faces)))]
        # for (i, (txt, val)) in enumerate(info):
        #     text = '{}: {}'.format(txt, val)
        #     cv2.putText(frame, text, (10, (i * 20) + 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)


def yoloface(args):
    if args.device == 'CPU':
        return Yoloface_CPU(args)
    elif args.device == 'GPU':
        return Yoloface_GPU(args)
yoloface.devices = ['CPU', 'GPU']
