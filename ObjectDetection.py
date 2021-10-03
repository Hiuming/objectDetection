import numpy as np
import argparse
import time
import cv2
import os

class ObjectDetection:
    confthres = 0.5
    nmsthres = 0.1
    yolo_path = "./"
    labelsPath = "./coco.names"
    cfgpath = "cfg/yolov3.cfg"
    wpath = "cfg/yolov3.weights"
    net = None
    LABELS = None
    CFG = None
    COLORS = None

    def __init__(self):
        self.LABELS = self.get_labels(self.yolo_path, self.labelsPath)
        self.CFG = self.get_config(self.yolo_path,self.cfgpath)
        self.Weights = self.get_weights(self.yolo_path,self.wpath)
        self.net = self.load_model(self.CFG, self.Weights)
        self.COLORS = self.get_colors(self.LABELS)

    @staticmethod
    def get_labels(yolo_path,labelsPath):
        lpath = os.path.sep.join([yolo_path, labelsPath])
        LABELS = open(lpath).read().strip().split("\n")
        return LABELS

    @staticmethod
    def get_colors(LABELS):
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        return COLORS


    @staticmethod
    def get_weights(yolo_path,weights_path):
        weightsPath = os.path.sep.join([yolo_path, weights_path])
        return weightsPath

    @staticmethod
    def get_config( yolo_path,config_path):
        configPath = os.path.sep.join([yolo_path, config_path])
        return configPath


    @staticmethod
    def load_model(config_path,weights_path):
        print("---------------[INFO] loading YOLO from disk...-------------------")
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        print("-----------------[SUCCESS] model loaded !!! ---------------------------")
        return net


    @staticmethod
    def get_predection(self,image):
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                # print(scores)
                classID = np.argmax(scores)
                # print(classID)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confthres:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confthres,
                                self.nmsthres)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


    def runModel(self,imagine):
        self.Colors = self.get_colors(self.LABELS)
        res = self.get_predection(self,imagine)
        return res