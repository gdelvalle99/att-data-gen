import numpy as np
import cv2
import pandas as pd
import os
import dlib
import csv
import time
from natsort import natsorted

start = time.time()

found = 0
not_found = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
def find_facial_landmarks_opencv(img, name):
    print("On", name)
    global found
    global not_found
    shape = None
    global detector
    global predictor
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray,1)
    #print(img)
    #path = os.getcwd()
    #rects = None
    #if no face is detected the picture goes to this directory
    if rects is None:
        not_found += 1
        shape = None
        if not os.path.exists("not_detected"):
            os.mkdir("not_detected")
        cv2.imwrite(os.path.join('not_detected/'+name), img)
    else:
        if not os.path.exists("detected_opencv"):
            os.mkdir("detected_opencv")
        cv2.imwrite(os.path.join('detected_opencv/'+name), img)
        found += 1
        for (i,rect) in enumerate(rects):
            shape = predictor(img_gray, rect)
            shape = shape_to_np(shape)
            (x,y,w,h) = rect_to_bb(rect)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0))

            for(x,y) in shape:
                cv2.circle(img, (x,y), 1, (0,0,255),-1)


            #returns marked image, and coordinates of the landmarks
    return img, shape


#finished
def extract_landmarks(name,shape, df):

    list = []
    for i in shape:
        list.append(i)
    df_length = len(df)
    df.loc[df_length]= list
    df.index = df.index[:-1].tolist() + [name]


    return df

#works on entire directories
def process_directory(dir, csv_file):
    list = []
    for i in range(68): #this section creates the columns for the csv file (hardcoded to work with 68 predictor)
        list.append(("x_"+str(i),"y_"+str(i)))
    df = pd.DataFrame(columns=[col for col in list])
    entries = natsorted(os.listdir(dir))
    for entry in entries:
        img = cv2.imread(dir+"/"+entry)
        if img is not None:
            img, shape = find_facial_landmarks_opencv(img, entry)
            if shape is not None:
                df = extract_landmarks(entry,shape, df)
    #print(df)
    df.to_csv(csv_file)
    return

csv_file = 'test.csv'
path = '/Users/guillermodelvalle/Desktop/celeba-dataset-2/img_align_celeba/img_align_celeba'
process_directory(path, csv_file)
print("Percentage of found:", found/(not_found+found))
end = time.time()
print("Time taken:", end - start)
