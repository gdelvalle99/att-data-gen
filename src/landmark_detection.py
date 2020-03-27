import numpy as np
import cv2
import pandas as pd
import os
import dlib
import csv
import time
from natsort import natsorted
import subprocess

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


#gets bbox and inserts it into a dictionary
def use_bbox(dir):
    df = pd.read_csv(dir)
    dict = {}
    for x in df.index:
        #saves values as tuples
        value = (df.at[x,"x_1"], df.at[x,"y_1"], df.at[x,"width"], df.at[x, "height"])
        #uses image name as the dictionary key
        dict[df.at[x,"image_id"]] = value
    return dict

def crop_images(dir, rect):
    entries = natsorted(os.listdir(dir))
    for entry in entries:
        img = cv2.imread(dir+"/"+entry)
        crop_img = img[dict[entry][1]:dict[entry][1]+dict[entry][3],dict[entry][0]:dict[entry][0]+dict[entry][2]]
        crop_img = cv2.resize(crop_img,(178,218))
    return

#finished
def extract_landmarks(name,shape, df):

    list = []
    for i in shape:
        list.append(i)
    #appends x and y coordinates as a single tuple
    df_length = len(df)
    df.loc[df_length]= list
    df.index = df.index[:-1].tolist() + [name]


    return df

def get_rect_OpenCV(rects, bbox):
    #print(len(rects))
    if(len(rects) == 1):
        return rects[0]
    else:
        (x1, y1, w1, h1) = bbox
        celebA_bbox = np.array((x1,y1))
        dist = float("inf")
        closest_bbox = None
        print(rects)
        for i,rect in enumerate(rects):
            print(rect)
            (x, y, w, h) = rect_to_bb(rect)
            dlib_bbox = np.array((x,y))
            #print(celebA_bbox - dlib_bbox)
            if np.linalg.norm(celebA_bbox - dlib_bbox) < dist:
                closest_bbox = rect
                dist = np.linalg.norm(celebA_bbox - dlib_bbox)
        #print(closest_bbox)
        return closest_bbox

def get_rect_OpenFace(rects, bbox):
    #print(len(rects))
    if(len(rects) == 1):
        return rects[0]
    else:
        (x1, y1, w1, h1) = bbox
        celebA_bbox = np.array((x1,y1))
        dist = float("inf")
        closest_bbox = None
        print(rects)
        for i,rect in enumerate(rects):
            print(rect)
            (x, y, w, h) = rect_to_bb(rect)
            dlib_bbox = np.array((x,y))
            #print(celebA_bbox - dlib_bbox)
            if np.linalg.norm(celebA_bbox - dlib_bbox) < dist:
                closest_bbox = rect
                dist = np.linalg.norm(celebA_bbox - dlib_bbox)
        #print(closest_bbox)
        return closest_bbox

#works on entire directories
def process_directory(dir, csv_file, dict):
    if not os.path.exists("detected_opencv"):
        os.mkdir("detected_opencv")
    if not os.path.exists("not_detected"):
        os.mkdir("not_detected")
    list = []
    global detector
    global predictor
    global found
    global not_found
    for i in range(68): #this section creates the columns for the csv file (hardcoded to work with 68 predictor)
        list.append(("x_"+str(i),"y_"+str(i)))
    df = pd.DataFrame(columns=[col for col in list])
    entries = natsorted(os.listdir(dir))
    for entry in entries:
        img = cv2.imread(dir+"/"+entry)
        if img is not None:
            shape = None
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(img_gray,1)
            #if no face is detected the picture goes to this directory
            #print(rects)
            if len(rects) == 0:
                not_found += 1
                shape = None

                cv2.imwrite(os.path.join('not_detected/'+entry), img)
            #if a face is detected then we go through the cropping process
            else:

                found += 1
                #this part identifies the features using dlib's face predictor
                rect = get_rect(rects, dict[entry])
                #here, the corners of the cropping square are identified
                left = rects[0].left()
                right = rects[0].right()
                top = rects[0].top()
                bottom = rects[0].bottom()
                #here we check if the cropping square goes out of bounds, and if it does, we set the coordinates to 0. (might not be needed)
                if(left < 0):
                    left = 0
                if(top < 0):
                    top = 0
                crop_img = img[top:bottom,left:right]
                crop_img = cv2.resize(crop_img,(178,218))
                cv2.imwrite(os.path.join('detected_opencv/'+entry), crop_img)
                shape = predictor(img_gray, rect)
                shape = shape_to_np(shape)
                (x,y,w,h) = rect_to_bb(rect)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0))

                for(x,y) in shape:
                    cv2.circle(img, (x,y), 1, (0,0,255),-1)
            if shape is not None:
                list = []
                for i in shape:
                    list.append(i)
                #appends x and y coordinates as a single tuple
                df_length = len(df)
                df.loc[df_length]= list
                df.index = df.index[:-1].tolist() + [entry]
                #df = extract_landmarks(entry,shape, df)
    df.to_csv(csv_file)
    return

def process_directory_openface(dir, csv_file, dict):
    if not os.path.exists("OpenFace_landmarks"):
        os.mkdir("OpenFace_landmarks")
    if not os.path.exists("not_detected"):
        os.mkdir("not_detected")
    OpenFaceBashCommand = '/home/guillermodelvalle/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -fdir '+dir+' -out_dir ../OpenFace_landmarks/'
    subprocess.call(OpenFaceBashCommand.split())
    list = []
    global detector
    global predictor
    global found
    global not_found
    for i in range(68): #this section creates the columns for the csv file (hardcoded to work with 68 predictor)
        list.append(("x_"+str(i),"y_"+str(i)))
    df = pd.DataFrame(columns=[col for col in list])

    return

dict = use_bbox('list_bbox_celeba.csv')
#print(dict['000001.jpg'])
csv_file = 'test.csv'
path = '/home/guillermodelvalle/img_celeba'
OpenFaceBashCommand = '/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -fdir '+path+' -out_dir ../OpenFace_landmarks/'
#print(OpenFaceBashCommand)
#process_directory(path, csv_file, dict)
process_directory_openface(path, csv_file, dict)
#print("Percentage of found:", found/(not_found+found))
#end = time.time()
#print("Time taken:", end - start)
