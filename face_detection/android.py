# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
u"""BlazeFace face detection.

Ported from Google® MediaPipe (https://google.github.io/mediapipe/).

Model card:

    https://mediapipe.page.link/blazeface-mc

Reference:

    V. Bazarevsky et al. BlazeFace: Sub-millisecond
    Neural Face Detection on Mobile GPUs. CVPR
    Workshop on Computer Vision for Augmented and
    Virtual Reality, Long Beach, CA, USA, 2019.
"""
import numpy as np
import os
from enum import IntEnum
from PIL.Image import Image
from typing import List, Optional, Union
from .nms import non_maximum_suppression
from .transform import detection_letterbox_removal, image_to_tensor
from .transform import sigmoid
from .types import Detection, Rect

MODEL_NAME_BACK = 'face_detection_back.tflite'
MODEL_NAME_FRONT = 'face_detection_front.tflite'
# score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
# this lower limit is safe for use with the sigmoid functions and float32
RAW_SCORE_LIMIT = 80
# threshold for confidence scores
MIN_SCORE = 0.5
# NMS similarity threshold
MIN_SUPPRESSION_THRESHOLD = 0.3

# from mediapipe module; irrelevant parts removed
# (reference: mediapipe/modules/face_detection/face_detection_front_cpu.pbtxt)
SSD_OPTIONS_FRONT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
}

# (reference: modules/face_detection/face_detection_back_desktop_live.pbtxt)
SSD_OPTIONS_BACK = {
    'num_layers': 4,
    'input_size_height': 256,
    'input_size_width': 256,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [16, 32, 32, 32],
}


class FaceIndex(IntEnum):
    """Indexes of keypoints returned by the face detection model.

    Use these with detection results (by indexing the result):
    ```
        def get_left_eye_position(detection):
            x, y = detection[FaceIndex.LEFT_EYE]
            return x, y
    ```
    """
    LEFT_EYE = 0
    RIGHT_EYE = 1
    NOSE_TIP = 2
    MOUTH = 3
    LEFT_EYE_TRAGION = 4
    RIGHT_EYE_TRAGION = 5


class FaceDetectionModel(IntEnum):
    """Face detection model option:

    FRONT_CAMERA - 128x128 image, assumed to be mirrored

    BACK_CAMERA - 256x256 image, not mirrored
    """
    FRONT_CAMERA = 0
    BACK_CAMERA = 1


from jnius import autoclass

File = autoclass('java.io.File')
Interpreter = autoclass('org.tensorflow.lite.Interpreter')
InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
Tensor = autoclass('org.tensorflow.lite.Tensor')
DataType = autoclass('org.tensorflow.lite.DataType')
TensorBuffer = autoclass(
    'org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
ByteBuffer = autoclass('java.nio.ByteBuffer')
jMap = autoclass('java.util.HashMap')

def dict_to_java_Map(data: dict):
    map = jMap()
    for key, value in data.items():
        map.put(key, value)
    return map

class FaceDetection(object):
    """BlazeFace face detection model as used by Google MediaPipe.

    This model can detect multiple faces and returns a list of detections.
    Each detection contains the normalised [0,1] position and size of the
    detected face, as well as a number of keypoints (also normalised to
    [0,1]).

    The model is callable and accepts a PIL image instance, image file name,
    and Numpy array of shape (height, width, channels) as input. There is no
    size restriction, but smaller images are processed faster.

    Example:

    ```
        detect_faces = FaceDetection(model_path='/var/mediapipe/models')
        detections = detect_faces('/home/user/pictures/group_photo.jpg')
        print(f'num. faces found: {len(detections)}')
        # convert normalised coordinates to pixels (assuming 3kx2k image):
        if len(detections) > 0:
            rect = detections[0].bbox.scale(3000, 2000)
            print(f'first face rect.: {rect}')
        else:
            print('no faces found')
    ```
    """
    def __init__(
        self,
        model_type: FaceDetectionModel = FaceDetectionModel.FRONT_CAMERA,
        model_path: Optional[str] = None,
        num_threads = None
    ) -> None:
        ssd_opts = {}
        if model_path is None:
            model_path = 'face_detection'
        if model_type == FaceDetectionModel.FRONT_CAMERA:
            # self.model_path = os.path.join(model_path, MODEL_NAME_FRONT)
            self.model_path = os.path.join(os.getcwd(),'face_detection', 'data', 'face_detection_front.tflite')
            ssd_opts = SSD_OPTIONS_FRONT
        elif model_type == FaceDetectionModel.BACK_CAMERA:
            self.model_path = os.path.join(model_path, MODEL_NAME_BACK)
            ssd_opts = SSD_OPTIONS_BACK
        else:
            raise ValueError(f'unsupported model_type "{model_type}"')
        model_path = File(self.model_path)
        
        options = InterpreterOptions()
        if num_threads is not None:
            options.setNumThreads(num_threads)

        self.interpreter = Interpreter(model_path, options)
        self.interpreter.allocateTensors()
        
        self.input_shape = self.interpreter.getInputTensor(0).shape()
        
        # try:
        self.output_indices = [0, 1] 
        # except:
        #     raise ZeroDivisionError(str(dir(self.interpreter.getOutputTensor(0))))
        self.output_shapes = [self.interpreter.getOutputTensor(i).shape() for i in range(2)] 
        
        self.output_type = self.interpreter.getOutputTensor(0).dataType() # same dtype for both outputs
        
        self.anchors = _ssd_generate_anchors(ssd_opts)

    def __call__(
        self,
        image: Union[Image, np.ndarray, str],
        roi: Optional[Rect] = None
    ) -> List[Detection]:
        """Run inference and return detections from a given image

        Args:
            image (Image|ndarray|str): Numpy array of shape
                `(height, width, 3)`, PIL Image instance or file name.

            roi (Rect|None): Optional region within the image that may
                contain faces.

        Returns:
            (list) List of detection results with relative coordinates.
        """
        height, width = self.input_shape[1:3]
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=True,
            output_range=(-1, 1))
        input_data = image_data.tensor_data[np.newaxis]
        raw_boxes, raw_scores = self._forward(input_data)
        boxes = self._decode_boxes(raw_boxes)
        scores = self._get_sigmoid_scores(raw_scores)
        detections = FaceDetection._convert_to_detections(boxes, scores)
        pruned_detections = non_maximum_suppression(
                                detections,
                                MIN_SUPPRESSION_THRESHOLD, MIN_SCORE,
                                weighted=True)
        detections = detection_letterbox_removal(
            pruned_detections, image_data.padding)
        return detections
    
    def _forward(self, X):
        input = ByteBuffer.wrap(X.tobytes())
        outputs = {}
        for index, shape in zip(self.output_indices, self.output_shapes):
            output = TensorBuffer.createFixedSize(shape,
                                                  self.output_type)
            outputs[index] = output
        output_data = {k: v.getBuffer().rewind() for k, v in outputs.items()}
        output_data = dict_to_java_Map(output_data)
        self.interpreter.runForMultipleInputsOutputs([input], output_data)
        outputs = [outputs[i] for i in self.output_indices]
        
        raw_boxes, raw_scores = [np.reshape(np.array(o.getFloatArray()), shape) for o, shape in zip(outputs, self.output_shapes)]
        
        return raw_boxes, raw_scores
        
    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Simplified version of
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # width == height so scale is the same across the board
        scale = self.input_shape[1]
        num_points = raw_boxes.shape[-1] // 2
        # scale all values (applies to positions, width, and height alike)
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale
        # adjust center coordinates and key points to anchor positions
        boxes[:, 0] += self.anchors
        for i in range(2, num_points):
            boxes[:, i] += self.anchors
        # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
        center = np.array(boxes[:, 0])
        half_size = boxes[:, 1] / 2
        boxes[:, 0] = center - half_size
        boxes[:, 1] = center + half_size
        return boxes

    def _get_sigmoid_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Extracted loop from ProcessCPU (line 327) in
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # just a single class ("face"), which simplifies this a lot
        # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
        #    causes overflow with IEEE single precision floats (max ~10e38)
        raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
        raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
        # 2) apply sigmoid function on clipped confidence scores
        return sigmoid(raw_scores)

    @staticmethod
    def _convert_to_detections(
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> List[Detection]:
        """Apply detection threshold, filter invalid boxes and return
        detection instance.
        """
        # return whether width and height are positive
        def is_valid(box: np.ndarray) -> bool:
            return np.all(box[1] > box[0])

        score_above_threshold = scores > MIN_SCORE
        filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
        filtered_scores = scores[score_above_threshold]
        return [Detection(box, score)
                for box, score in zip(filtered_boxes, filtered_scores)
                if is_valid(box)]


def _ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            repeats += 2    # aspect_ratios are added twice per iteration
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)
