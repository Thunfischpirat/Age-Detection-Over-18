# Forked from https://github.com/smahesh29/Gender-and-Age-Detection

import cv2
import argparse

import numpy as np


def highlight_face(net, frame, conf_threshold=0.7):
    frame_ddn=frame.copy()
    frame_height=frame_ddn.shape[0]
    frame_width=frame_ddn.shape[1]
    blob=cv2.dnn.blobFromImage(frame_ddn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    face_boxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3] * frame_width)
            y1=int(detections[0,0,i,4] * frame_height)
            x2=int(detections[0,0,i,5] * frame_width)
            y2=int(detections[0,0,i,6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes


parser=argparse.ArgumentParser()
parser.add_argument('--source')
parser.add_argument('--one_face', action='store_true')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=["(Under 18)", "(Check Id)", "(Over 18)"]

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)


video=cv2.VideoCapture(args.source if args.source else 0)
padding=20
while cv2.waitKey(1) == -1 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    faceBoxes = highlight_face(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        continue

    if args.one_face and len(faceBoxes) > 1:
        print("Only one person at a time allowed")
        continue

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        ageNet.setInput(blob)
        age_preds=ageNet.forward()
        under18_probs = np.sum(age_preds[:,:3], keepdims=True)
        checkid_probs = np.sum(age_preds[:,3:5], keepdims=True)
        over18_probs = np.sum(age_preds[:,5:], keepdims=True)
        ageBrackets = np.concatenate((under18_probs, checkid_probs, over18_probs), axis=1)
        age=ageList[ageBrackets.argmax()]

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        x1,y1,x2,y2 = faceBox

        if age == "(Under 18)":
            color = (0, 0, 255) # red
        elif age == "(Check Id)":
            color = (0, 255, 255) # yellow
        else:
            color = (0, 255, 0) # green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, frameHeight // 150, 8)
        cv2.imshow("Detecting age", frame)
