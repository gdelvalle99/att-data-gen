import numpy as np
import cv2
import pandas as pd
import os
import dlib

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x,y,w,h)

def shape_to_np(shape, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords



def find_facial_landmarks(img, name):
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray,1)
    #print(img)
    #path = os.getcwd()
    #rects = None
    if rects is None:
        if not os.path.exists("not_detected"):
            os.mkdir("not_detected")
        cv2.imwrite(os.path.join('not_detected/'+name), img)
    else:
        for (i,rect) in enumerate(rects):
            shape = predictor(img_gray, rect)
            shape = shape_to_np(shape)
            (x,y,w,h) = rect_to_bb(rect)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0))

            for(x,y) in shape:
                cv2.circle(img, (x,y), 1, (0,0,255),-1)



    return img



img = cv2.imread("000001.jpg")
img = find_facial_landmarks(img, "000001.jpg")
cv2.imshow("test", img)
cv2.waitKey(0)
