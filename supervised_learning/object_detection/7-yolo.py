#!/usr/bin/env python3
"""File that contains the class Yolo"""
import keras as K
import numpy as np
import cv2
import glob
import os
from google.colab.patches import cv2_imshow


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
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_h in range(pred.shape[0]):
                for grid_w in range(pred.shape[1]):
                    bx = ((self.sigmoid(pred[grid_h,
                                        grid_w, :,
                                        0]) + grid_w) / pred.shape[1])
                    by = ((self.sigmoid(pred[grid_h,
                                        grid_w, :,
                                        1]) + grid_h) / pred.shape[0])
                    anchor_tensor = self.anchors[ipred].astype(float)
                    anchor_tensor[:, 0] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               2]) / self.model.input.shape[1]  # bw
                    anchor_tensor[:, 1] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               3]) / self.model.input.shape[2]  # bh

                    pred[grid_h, grid_w, :, 0] = \
                        (bx - (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]  # x1
                    pred[grid_h, grid_w, :, 1] = \
                        (by - (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]  # y1
                    pred[grid_h, grid_w, :, 2] = \
                        (bx + (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]  # x2
                    pred[grid_h, grid_w, :, 3] = \
                        (by + (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]  # y2
        
        # box confidence
        box_confidences = [self.sigmoid(pred[:, :, :, 4:5]) for pred in outputs]

        # box class probs
        box_class_probs = [self.sigmoid(pred[:, :, :,5:]) for pred in outputs]
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

    def preprocess_images(self, images):
        """
        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
          pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3) containing all of the preprocessed images
            ni: the number of images that were preprocessed
            input_h: the input height for the Darknet model Note: this can vary by model
            input_w: the input width for the Darknet model Note: this can vary by model
            3: number of color channels
          image_shapes: a numpy.ndarray of shape (ni, 2) containing the original height and width of the images
            2 => (image_height, image_width)
        """
        p_images = []
        shapes = []

        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        for i in images:
            image_shape = i.shape[0], i.shape[1]
            shapes.append(image_shape)

            image = cv2.resize(i, (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255
            p_images.append(image)

        pimages = np.array(p_images)
        image_shapes = np.array(shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored
        Displays the image with all boundary boxes, class names, and box scores (see example below)
          Boxes should be drawn as with a blue line of thickness 2
          Class names and box scores should be drawn above each box in red
            Box scores should be rounded to 2 decimal places
            Text should be written 5 pixels above the top left corner of the box
            Text should be written in FONT_HERSHEY_SIMPLEX
            Font scale should be 0.5
            Line thickness should be 1
            You should use LINE_AA as the line type
          The window name should be the same as file_name
          If the s key is pressed:
            The image should be saved in the directory detections, located in the current directory
            If detections does not exist, create it
            The saved image should have the file name file_name
            The image window should be closed
          If any key besides s is pressed, the image window should be closed without saving
        """
        for i, box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])

            start_point = x1, y1
            end_point = int(box[2]), int(box[3])

            scores = f"{(box_scores[i])}"
            label = f"{self.class_names[box_classes[i]]}: {scores}"
            oorg = (x1, y1 - 5)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            text_color = (0, 0, 225)
            thick = 1
            line_type = cv2.LINE_AA

            image = cv2.rectangle(image, start_point,
                                  end_point, (255, 0, 0), thickness=2)
            print(image)
            image = cv2.putText(image, label, oorg, font, scale, text_color, thick,
                                line_type, bottomLeftOrigin=False)

        cv2_imshow(image)

        k = cv2.waitKey(0)
        if k == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            os.chdir('detections')
            cv2.imwrite(file_name, image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict function
        Args: folder_path: a string representing the path to the folder
              holding all the images to predict
        Returns: tuple of (predictions, image_paths)
        """
        predictions = []
        images, image_paths = self.load_images(folder_path)
        pimage, image_shape = self.preprocess_images(images)
        output_image = self.model.predict(pimage)

        for i, img in enumerate(images):
            outputs = [out[i] for out in output_image]
            bx, bclass, bscore = self.process_outputs(outputs, image_shape[i])
            bx, bclass, bscore = self.filter_boxes(bx, bclass, bscore)
            bx, bclass, bscore = self.non_max_suppression(bx, bclass,
                                                          bscore)
            predictions.append((bx, bclass, bscore))
            name = image_paths[i].split("/")[-1]
            self.show_boxes(img, bx, bclass, bscore, name)
        return predictions, image_paths