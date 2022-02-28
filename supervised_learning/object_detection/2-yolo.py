#!/usr/bin/env python3
"""File that contains the class Yolo"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Class YOLO(You only look once)"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored.
        classes_path is the path to where the list of class names used for
        the Darknet model, listed in order of index, can be found.
        class_t is a float representing the box score threshold for the initial filtering step.
        nms_t is a float representing the IOU threshold for non-max suppression.
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2) containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        model = K.models.load_model(model_path)

        with open(classes_path, "r") as classes_file:
            classes = [line.strip() for line in classes_file.readlines()]

        self.model = model
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process the outputs"""
        boxes = [output[:, :, :, 0:4] for output in outputs]

        for output_idx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    center_y = sigmoid(-output[y, x, :, 1]) + y
                    center_x = sigmoid(-output[y, x, :, 1]) + x

                    prior_resizes = self.anchors[output_idx].astype(float)
                    prior_resizes[:, 0] *= (np.exp(output[y, x, :, 2])
                                            / 2 * image_size[1] /
                                            self.model.input.shape[1])
                    prior_resizes[:, 1] *= (np.exp(output[y, x, :, 3])
                                            / 2 * image_size[0] /
                                            self.model.input.shape[2])

                    output[y, x, :, 0] = center_x - prior_resizes[:, 0]
                    output[y, x, :, 1] = center_y - prior_resizes[:, 1]
                    output[y, x, :, 2] = center_x + prior_resizes[:, 0]
                    output[y, x, :, 3] = center_y + prior_resizes[:, 1]

        box_confidences = [1 / (1 + np.exp(-output[:, :, :, 4, np.newaxis]))
                for output in outputs]
        box_class_probs = [1 / (1 + np.exp(-output[:, :, :, 5:]))
                           for output in outputs]

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 4)
        containing the processed boundary boxes for each output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 1)
        containing the processed box confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, classes)
        containing the processed box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class number that each box in filtered_boxes
            predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores for
            each box in filtered_boxes, respectively
        """
        all_boxes = np.concatenate([boxs.reshape(-1, 4) for boxs in boxes])
        class_probs = np.concatenate([probs.reshape(-1, box_class_probs[0].shape[-1])
                                      for probs in box_class_probs])
        all_classes = class_probs.argmax(axis=1)
        all_confidences = (np.concatenate([conf.reshape(-1)
                                           for conf in box_confidences])
                           * class_probs.max(axis=1))
        
        thresh_idxs = np.where(all_confidences < self.class_t)
        
        return (np.delete(all_boxes, thresh_idxs, axis=0),
                np.delete(all_classes, thresh_idxs),
                np.delete(all_confidences, thresh_idxs))

def sigmoid(x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))                    
