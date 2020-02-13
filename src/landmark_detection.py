import numpy as np
import cv2
import pandas as pd
import os
import dlib
import csv


#imported from face_utils
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x,y,w,h)

#imported from face utils
def shape_to_np(shape, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords




#marks the face with opencv and dlib
def find_facial_landmarks(img, name):
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray,1)
    #print(img)
    #path = os.getcwd()
    #rects = None
    #if no face is detected the picture goes to this directory
    if rects is None:
        shape = None
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


            #returns marked image, and coordinates of the landmarks
    return img, shape


#in progress!!
def extract_landmarks(name,shape):

    with open('placeholder.csv','w', newline='') as out:
        csv_out=csv.writer(out)
        #csv_out.writerow(["features","coordinates","y"])
        print(shape)
        csv_out.writerow(["x_"+str(index) for index,row in enumerate(shape)])
        csv_out.writerow([row for row in shape])
        #for index,row in enumerate(shape):
            #csv_out.writerow(["x_"+str(index),row)

    return

#works on entire directories
def process_directory(dir):
    entries = os.listdir(dir)
    for entry in entries:
        img = cv2.imread(dir+"/"+entry)
        if img is not None:
            img, shape = find_facial_landmarks(img, entry)
            if shape is not None:
                extract_landmarks(entry,shape)

    return

path = os.getcwd()
process_directory(path)
