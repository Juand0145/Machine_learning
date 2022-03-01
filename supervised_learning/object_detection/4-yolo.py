#!/usr/bin/env python3
"""File that contains the class Yolo"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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
            2 => [anchor_box_width, anchor_box_hei

            ght]
        """
        model = K.models.load_model(model_path)

        with open(classes_path, "r") as classes_file:
            classes = [line.strip() for line in classes_file.readlines()]

        self.model = model
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process the outputs"""
        boxes = [output[:, :, :, 0:4] for output in outputs]

        for output_idx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    center_y = self.sigmoid(-output[y, x, :, 1]) + y
                    center_x = self.sigmoid(-output[y, x, :, 1]) + x

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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number for the
        class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for each box
        in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes, predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of the predicted
            bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing the class number
            for box_predictions ordered by class and box score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing the box scores for
            box_predictions ordered by class and box score, respectively
        """
        f = []
        c = []
        s = []

        for i in np.unique(box_classes):
            idx = np.where(box_classes == i)
            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]
            keep = self.nms(filters, self.nms_t, scores)

            filters = filters[keep]
            scores = scores[keep]
            classes = classes[keep]

            f.append(filters)
            c.append(classes)
            s.append(scores)

        filtered_boxes = np.concatenate(f, axis=0)
        box_scores = np.concatenate(c, axis=0)
        box_classes = np.concatenate(s, axis=0)

        return filtered_boxes, box_scores, box_classes

    def nms(self, bc, thresh, scores):
        """
        Function that computes the index
        Args:
            bc: Box coordinates
            thresh: Threeshold
            scores: scores for each box indexed and sorted
        Returns: Sorted index score for non max supression
        """
        x1 = bc[:, 0]
        y1 = bc[:, 1]
        x2 = bc[:, 2]
        y2 = bc[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def load_images(folder_path):
        """
        folder_path: a string representing the path to the folder holding all the images to load
        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(f"{folder_path}/*")

        images = [cv2.imread(i) for i in image_paths]

        return images, image_paths
