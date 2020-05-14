import numpy as np
import cv2
import pandas as pd
import os
import dlib
import csv
import time
from natsort import natsorted
import subprocess
#import point
import shutil

start = time.time()
print(start)
found = 0
not_found = 0
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#imported from face_utils
cropsize = (178,218)

def crop_openface(img,bbox,size,name):
    #bbox[24]
    #print(img)
    dimensions = img.shape

    top = (bbox[24][1] - (bbox[8][1]*1.1))
    if(top < 0 or top >img.shape[0]):
        top = 0
    #print(top)
    left = bbox[0][0] - (bbox[36][0]-bbox[0][0])
    #print(bbox[0][0])
    #print(bbox[0],bbox[16])
    if(left < 0 or left > img.shape[1]):
        left = 0
    right = bbox[16][0] + (bbox[16][0]-bbox[45][0])
    if(right > img.shape[1] or right < 0):
        right = img.shape[1]-1
    #print(bbox[16][0])
    bottom = bbox[8][1] - (bbox[33][1] - bbox[8][1])
    if(bottom > img.shape[0] or bottom < 0):
        bottom = img.shape[0]-1
    #right = bbox[17]
    #bottom = bbox[9]
    #print(top,left,right,bottom)
    if(left > right):
        temp = left
        left = right
        right = temp
    if(top > bottom):
        temp = top
        top = bottom
        bottom = temp
    crop = img[int(top):int(bottom), int(left):int(right)]
    #print(crop.shape)
    if(crop.shape[0] is 0 or crop.shape[1] is 0):
        cv2.imwrite("/home/guillermodelvalle/OpenFace_not_detected/"+name,img)
        return None

    crop = cv2.resize(crop,size)
    #crop = (crop > 0).astype(np.uint8) * 255
    return crop

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
    #print("On", name)
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
def extract_landmarks_opencv(name,shape, df):

    list = []
    for i in shape:
        list.append(i)
    #appends x and y coordinates as a single tuple
    df_length = len(df)
    df.loc[df_length]= list
    df.index = df.index[:-1].tolist() + [name]


    return df

global missed_count
missed_count = 0

def extract_landmarks_done(name,dir,df,file_name,dict,out):
    entry = dir + name[:-4] + ".csv"
    #print(entry)

    #print(file_name+name)
    filename = file_name+'/' + name
    #print(filename)
        #print(file_name)
        #print(filename)
    #shutil.copy(filename,file_name+"/OpenFace_detected")
    img = cv2.imread(filename)
    list = []
    old_df = pd.read_csv(entry)
    k = get_rect_OpenFace(old_df,dict)
    for i in range(68):
       # print(old_df.at[k," x_"+str(i)], old_df.at[k," y_"+str(i)])
        value = (old_df.at[k," x_"+str(i)], old_df.at[k," y_"+str(i)])
        list.append(value)

    df_length = len(df)
       # print(list)
    df.loc[df_length] = list
    df.index = df.index[:-1].tolist() + [name]
    #print(filename)

    #print(file_name+'OpenFace_detected/'+name)
    cv2.imwrite(out+"/"+name,img)
        #print(out+"/"+name)
    return df



def extract_landmarks_openface(name,dir,df,file_name,dict,out):
    entry = dir + name[:-4] + ".csv"
    #print(entry)
    if(os.path.isfile(entry) is False):
        if(os.path.isdir(file_name+name) is True):
            return df
        filename = file_name+'/'+name
        #print(file_name)
        global missed_count
        missed_count += 1
        #print(filename)
        shutil.copy(filename,"/home/guillermodelvalle/OpenFace_not_detected")
        return df
    else:
        #print(file_name+name)
        filename = file_name+'/' + name
        #print(filename)
        #print(file_name)
        #print(filename)
        #shutil.copy(filename,file_name+"/OpenFace_detected")
        img = cv2.imread(filename)
        list = []
        old_df = pd.read_csv(entry)
        k = get_rect_OpenFace(old_df,dict)
        for i in range(68):
           # print(old_df.at[k," x_"+str(i)], old_df.at[k," y_"+str(i)])
            value = (old_df.at[k," x_"+str(i)], old_df.at[k," y_"+str(i)])
            list.append(value)

        img = crop_openface(img,list,cropsize,name)
        if img is None:
            return df

        df_length = len(df)
       # print(list)
        df.loc[df_length] = list
        df.index = df.index[:-1].tolist() + [name]
        #print(filename)

        #print(file_name+'OpenFace_detected/'+name)
        cv2.imwrite(out+"/"+name,img)
        #print(out+"/"+name)
        return df


def get_rect(rects, bbox):
    #print(len(rects))
    if(len(rects) == 1):
        return rects[0]
    else:
        (x1, y1, w1, h1) = bbox
        celebA_bbox = np.array((x1,y1))
        dist = float("inf")
        closest_bbox = None
      #  print(rects)
        for i,rect in enumerate(rects):
           # print(rect)
            (x, y, w, h) = rect_to_bb(rect)
            dlib_bbox = np.array((x,y))
            #print(celebA_bbox - dlib_bbox)
            if np.linalg.norm(celebA_bbox - dlib_bbox) < dist:
                closest_bbox = rect
                dist = np.linalg.norm(celebA_bbox - dlib_bbox)
        #print(closest_bbox)
        return closest_bbox

def get_rect_OpenFace(of_landmarks, bbox):
    #print(len(rects))
    if(len(of_landmarks.index) == 1):
        return 0
    else:
        (x1, y1, w1, h1) = bbox
        celebA_bbox = np.array((x1,y1))
        dist = float("inf")
        closest_bbox = None
     #   print(rects)
    #    print(of_landmarks.index)
        #print(list(of_landmarks[" y_0"]))
        for i in of_landmarks.index:
            coords = np.array(((int(round(of_landmarks.iloc[i][2]))), (int(round(of_landmarks.iloc[i][70])))))
            ##dlib_bbox = np.array((x,y))
            #print(celebA_bbox - dlib_bbox)
            #print(coords)
            if np.linalg.norm(celebA_bbox - coords) < dist:
                closest_bbox = i
                dist = np.linalg.norm(celebA_bbox - coords)
        #print(closest_bbox)
        return closest_bbox

#works on entire directories
def process_directory(dir, csv_file, dict):
    if not os.path.exists("/home/guillermodelvalle/detected_opencv"):
        os.mkdir("/home/guillermodelvalle/detected_opencv")
    if not os.path.exists("/home/guillermodelvalle/not_detected"):
        os.mkdir("/home/guillermodelvalle/not_detected")
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

                cv2.imwrite(os.path.join('/home/guillermodelvalle/not_detected/'+entry), img)
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
                #crop_img = img[top:bottom,left:right]
                #crop_img = cv2.resize(crop_img,(178,218))




            if shape is not None:
                cv2.imwrite(os.path.join('/home/guillermodelvalle/detected_opencv/'+entry), crop_img)
                shape = predictor(img_gray, rect)
                shape = shape_to_np(shape)
                crop_img = crop_openface(img,shape,cropsize,entry)
                if(crop_img is None):
                    cv2.imwrite(os.path.join('/home/guillermodelvalle/not_detected/'+entry), img)

                else:
                    (x,y,w,h) = rect_to_bb(rect)
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0))
                    list = []
                    for i in shape:
                        list.append(i)
                #appends x and y coordinates as a single tuple
                    df_length = len(df)
                    df.loc[df_length]= list
                    df.index = df.index[:-1].tolist() + [entry]
                #df = extract_landmarks(entry,shape, df)
    return df

def process_directory_openface(dir, csv_file, dict):
    if not os.path.exists("/home/guillermodelvalle/OpenFace_detected"):
        os.mkdir("/home/guillermodelvalle/OpenFace_detected")
    if not os.path.exists("/home/guillermodelvalle/OpenFace_not_detected"):
        os.mkdir("/home/guillermodelvalle/OpenFace_not_detected")
    if not os.path.exists("/home/guillermodelvalle/OpenFace_landmarks"):
        os.mkdir("/home/guillermodelvalle/OpenFace_landmarks")
    OpenFaceBashCommand = '/home/guillermodelvalle/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -fdir '+dir+' -out_dir /home/guillermodelvalle/OpenFace_landmarks'
   # subprocess.call(OpenFaceBashCommand.split())
    list = []
    global detector
    global predictor
    global found
    global not_found
    for i in range(68): #this section creates the columns for the csv file (hardcoded to work with 68 predictor)
        list.append(("x_"+str(i),"y_"+str(i)))
    df = pd.DataFrame(columns=[col for col in list])
    entries = natsorted(os.listdir('/home/guillermodelvalle/OpenFace_detected'))
    j = 0
    for entry in entries:
        j+=1
        if(j % 5000 is 0):
            print(entry)
        df = extract_landmarks_done(entry,'/home/guillermodelvalle/OpenFace_landmarks/',df,dir,dict[entry],"/home/guillermodelvalle/OpenFace_detected")
    return df


def create_new_csv(df, csv_file):
    old_df = pd.read_csv(csv_file)
    labels = old_df.columns.values
    rows = df.index.values
    #print(old_df.index.values)
    #print(old_df.loc[0])
    newdf = pd.DataFrame(columns=labels)#,index=rows)
    #print(newdf)
    for i in range(0,len(df)):
       # df_length = len(newdf)
        name = df.index[i]
        name = int(name[:-4]) - 1
        row = old_df.loc[name]
        #print(row)
        newdf.loc[name]= row
        #print(newdf.iloc[i])
        #newdf.index = df.index[:-1].tolist() + [name]
    #entries = natsorted(os.listdir(dir))
    #for entry in entries:
    #    img = cv2.imread(dir+"/"+entry)


    return newdf




def five_oclock_shadow(img_n,df):
    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)
    points = [chin, right_cheek, left_cheek, neck]##neck
    return points

def arched_eyebrows(img_n,df):
    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)
    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    points = [left_eyebrow,right_eyebrow]
    return points

def attractive(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    upper_lip = np.array([df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
    df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
    df.iloc[img_n,49],df.iloc[img_n,48]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    left_eye = np.array([df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
    df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],dtype=np.int32)

    right_eye = np.array([df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
    df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]],dtype=np.int32)

    nose = np.array([[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
    df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    top_of_head = np.array([df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
    [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]],dtype=np.int32)

    left_ear = np.array([[df.iloc[img_n,0][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,0][0]),df.iloc[img_n,0][1]],
    [df.iloc[img_n,1][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,1][0]),df.iloc[img_n,1][1]],
    [df.iloc[img_n,2][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,2][0]),df.iloc[img_n,2][1]],
    [df.iloc[img_n,3][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,3][0]),df.iloc[img_n,3][1]],
    df.iloc[img_n,3],df.iloc[img_n,2],df.iloc[img_n,1],df.iloc[img_n,0]],dtype=np.int32)

    right_ear = np.array([[df.iloc[img_n,16][0]+(df.iloc[img_n,16][0]-df.iloc[img_n,26][0]),df.iloc[img_n,16][1]],
    [df.iloc[img_n,15][0]+(df.iloc[img_n,15][0]-df.iloc[img_n,26][0]),df.iloc[img_n,15][1]],
    [df.iloc[img_n,14][0]+(df.iloc[img_n,14][0]-df.iloc[img_n,26][0]),df.iloc[img_n,14][1]],
    [df.iloc[img_n,13][0]+(df.iloc[img_n,13][0]-df.iloc[img_n,26][0]),df.iloc[img_n,13][1]],
    df.iloc[img_n,13],df.iloc[img_n,14],df.iloc[img_n,15],df.iloc[img_n,16]],dtype=np.int32)


    points = [chin, left_cheek, right_cheek, upper_lip, neck, mouth, left_eye, right_eye, nose, left_eyebrow, right_eyebrow, top_of_head, left_ear, right_ear]

    return points

def bags_under_eyes(img_n,df):
    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    points = [left_cheek,right_cheek]
    return points

def bald(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def bangs(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3
    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def big_lips(img_n,df):
    points = np.array([[df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
        df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
        df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
        df.iloc[img_n,59],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def big_nose(img_n,df):
    points = np.array([[[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
        df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
        df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]]],dtype=np.int32)
    return points

def black_hair(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3
    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def blond_hair(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def blurry(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    upper_lip = np.array([df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
    df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
    df.iloc[img_n,49],df.iloc[img_n,48]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    left_eye = np.array([df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
    df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],dtype=np.int32)

    right_eye = np.array([df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
    df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]],dtype=np.int32)

    nose = np.array([[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
    df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    top_of_head = np.array([df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
    [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]],dtype=np.int32)

    left_ear = np.array([[df.iloc[img_n,0][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,0][0]),df.iloc[img_n,0][1]],
    [df.iloc[img_n,1][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,1][0]),df.iloc[img_n,1][1]],
    [df.iloc[img_n,2][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,2][0]),df.iloc[img_n,2][1]],
    [df.iloc[img_n,3][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,3][0]),df.iloc[img_n,3][1]],
    df.iloc[img_n,3],df.iloc[img_n,2],df.iloc[img_n,1],df.iloc[img_n,0]],dtype=np.int32)

    right_ear = np.array([[df.iloc[img_n,16][0]+(df.iloc[img_n,16][0]-df.iloc[img_n,26][0]),df.iloc[img_n,16][1]],
    [df.iloc[img_n,15][0]+(df.iloc[img_n,15][0]-df.iloc[img_n,26][0]),df.iloc[img_n,15][1]],
    [df.iloc[img_n,14][0]+(df.iloc[img_n,14][0]-df.iloc[img_n,26][0]),df.iloc[img_n,14][1]],
    [df.iloc[img_n,13][0]+(df.iloc[img_n,13][0]-df.iloc[img_n,26][0]),df.iloc[img_n,13][1]],
    df.iloc[img_n,13],df.iloc[img_n,14],df.iloc[img_n,15],df.iloc[img_n,16]],dtype=np.int32)


    points = [chin, left_cheek, right_cheek, upper_lip, neck, mouth, left_eye, right_eye, nose, left_eyebrow, right_eyebrow, top_of_head, left_ear, right_ear]
    return points

def brown_hair(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def bushy_eyebrows(img_n,df):
    points = np.array([[df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],[df.iloc[img_n,22],df.iloc[img_n,23],
        df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]]],dtype=np.int32)
    return points

def chubby(img_n,df):

    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    upper_lip = np.array([df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
    df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
    df.iloc[img_n,49],df.iloc[img_n,48]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    points = [chin, left_cheek, right_cheek, upper_lip, neck, mouth, left_eyebrow, right_eyebrow]## here are the eyebrows
    return points

def double_chin(img_n,df):
    points = np.array([[df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],
                        df.iloc[img_n,10],df.iloc[img_n,11],[df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])],
                        [df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])]]],dtype=np.int32)
    return points

def eyeglasses(img_n,df):

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    left_eye = np.array([df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
    df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],dtype=np.int32)

    right_eye = np.array([df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
    df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    left_ear = np.array([[df.iloc[img_n,0][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,0][0]),df.iloc[img_n,0][1]],
    [df.iloc[img_n,1][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,1][0]),df.iloc[img_n,1][1]],
    [df.iloc[img_n,2][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,2][0]),df.iloc[img_n,2][1]],
    [df.iloc[img_n,3][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,3][0]),df.iloc[img_n,3][1]],
    df.iloc[img_n,3],df.iloc[img_n,2],df.iloc[img_n,1],df.iloc[img_n,0]],dtype=np.int32)

    right_ear = np.array([[df.iloc[img_n,16][0]+(df.iloc[img_n,16][0]-df.iloc[img_n,26][0]),df.iloc[img_n,16][1]],
    [df.iloc[img_n,15][0]+(df.iloc[img_n,15][0]-df.iloc[img_n,26][0]),df.iloc[img_n,15][1]],
    [df.iloc[img_n,14][0]+(df.iloc[img_n,14][0]-df.iloc[img_n,26][0]),df.iloc[img_n,14][1]],
    [df.iloc[img_n,13][0]+(df.iloc[img_n,13][0]-df.iloc[img_n,26][0]),df.iloc[img_n,13][1]],
    df.iloc[img_n,13],df.iloc[img_n,14],df.iloc[img_n,15],df.iloc[img_n,16]],dtype=np.int32)


    points = [left_cheek, right_cheek, left_eye, right_eye, left_eyebrow, right_eyebrow, left_ear, right_ear]

    return points

def goatee(img_n,df):
    points = np.array([[df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],
        df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
        df.iloc[img_n,54],df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],
        df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def gray_hair(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3
    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def heavy_makeup(img_n,df):

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    left_eye = np.array([df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
    df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],dtype=np.int32)

    right_eye = np.array([df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
    df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    points = [left_cheek, right_cheek, mouth, left_eye, right_eye, left_eyebrow, right_eyebrow]
    return points

def high_cheekbones(img_n,df):
    points = np.array([[df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],[df.iloc[img_n,16],
        df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
        df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],
        df.iloc[img_n,45],df.iloc[img_n,16]]],dtype=np.int32)
    return points

def male(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    upper_lip = np.array([df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
    df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
    df.iloc[img_n,49],df.iloc[img_n,48]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    left_eye = np.array([df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
    df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],dtype=np.int32)

    right_eye = np.array([df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
    df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]],dtype=np.int32)

    nose = np.array([[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
    df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    top_of_head = np.array([df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
    [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]],dtype=np.int32)

    left_ear = np.array([[df.iloc[img_n,0][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,0][0]),df.iloc[img_n,0][1]],
    [df.iloc[img_n,1][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,1][0]),df.iloc[img_n,1][1]],
    [df.iloc[img_n,2][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,2][0]),df.iloc[img_n,2][1]],
    [df.iloc[img_n,3][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,3][0]),df.iloc[img_n,3][1]],
    df.iloc[img_n,3],df.iloc[img_n,2],df.iloc[img_n,1],df.iloc[img_n,0]],dtype=np.int32)

    right_ear = np.array([[df.iloc[img_n,16][0]+(df.iloc[img_n,16][0]-df.iloc[img_n,26][0]),df.iloc[img_n,16][1]],
    [df.iloc[img_n,15][0]+(df.iloc[img_n,15][0]-df.iloc[img_n,26][0]),df.iloc[img_n,15][1]],
    [df.iloc[img_n,14][0]+(df.iloc[img_n,14][0]-df.iloc[img_n,26][0]),df.iloc[img_n,14][1]],
    [df.iloc[img_n,13][0]+(df.iloc[img_n,13][0]-df.iloc[img_n,26][0]),df.iloc[img_n,13][1]],
    df.iloc[img_n,13],df.iloc[img_n,14],df.iloc[img_n,15],df.iloc[img_n,16]],dtype=np.int32)


    points = [chin, left_cheek, right_cheek, upper_lip, neck, mouth, left_eye, right_eye, nose, left_eyebrow, right_eyebrow, top_of_head, left_ear, right_ear]

    return points

def mouth_slightly_open(img_n,df):
    points = np.array([[df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
        df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
        df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
        df.iloc[img_n,59],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def mustache(img_n,df):
    points = np.array([[df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
        df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
        df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
        df.iloc[img_n,49],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def narrow_eyes(img_n,df):
    points = np.array([[df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
        df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],
        [df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
        df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]]],dtype=np.int32)
    return points

def no_beard(img_n,df):
    points = np.array([[df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],
                        df.iloc[img_n,10],df.iloc[img_n,11],[df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])],
                        [df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])]]],dtype=np.int32)
    return points

def oval_face(img_n,df):
    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    points = [chin, left_cheek, right_cheek]
    return points

def pale_skin(img_n,df):

    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    upper_lip = np.array([df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
    df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
    df.iloc[img_n,49],df.iloc[img_n,48]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    nose = np.array([[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
    df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]],dtype=np.int32)

    points = [chin, left_cheek, right_cheek, upper_lip, neck, mouth, nose]
    return points

def pointy_nose(img_n,df):
    points = np.array([[[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
        df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
        df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]]],dtype=np.int32)
    return points

def receding_hairline(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def rosy_cheeks(img_n,df):
    points = np.array([[df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],[df.iloc[img_n,16],
        df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
        df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],
        df.iloc[img_n,45],df.iloc[img_n,16]]],dtype=np.int32)
    return points

def sideburns(img_n,df):
    points = np.array([[df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],[df.iloc[img_n,16],
        df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
        df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],
        df.iloc[img_n,45],df.iloc[img_n,16]]],dtype=np.int32)
    return points

def smiling(img_n,df):
    points = np.array([[df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
        df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
        df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
        df.iloc[img_n,59],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def straight_hair(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def wavy_hair(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def wearing_earrings(img_n,df):
    points = np.array([[[df.iloc[img_n,0][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,0][0]),df.iloc[img_n,0][1]],
                        [df.iloc[img_n,1][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,1][0]),df.iloc[img_n,1][1]],
                        [df.iloc[img_n,2][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,2][0]),df.iloc[img_n,2][1]],
                        [df.iloc[img_n,3][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,3][0]),df.iloc[img_n,3][1]],
                        df.iloc[img_n,3],df.iloc[img_n,2],df.iloc[img_n,1],df.iloc[img_n,0]],
                        [[df.iloc[img_n,16][0]+(df.iloc[img_n,16][0]-df.iloc[img_n,26][0]),df.iloc[img_n,16][1]],
                        [df.iloc[img_n,15][0]+(df.iloc[img_n,15][0]-df.iloc[img_n,26][0]),df.iloc[img_n,15][1]],
                        [df.iloc[img_n,14][0]+(df.iloc[img_n,14][0]-df.iloc[img_n,26][0]),df.iloc[img_n,14][1]],
                        [df.iloc[img_n,13][0]+(df.iloc[img_n,13][0]-df.iloc[img_n,26][0]),df.iloc[img_n,13][1]],
                        df.iloc[img_n,13],df.iloc[img_n,14],df.iloc[img_n,15],df.iloc[img_n,16]]],dtype=np.int32)
    return points

def wearing_hat(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    points = np.array([[df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
        [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]]],dtype=np.int32)
    return points

def wearing_lipstick(img_n,df):
    points = np.array([[df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
        df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
        df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
        df.iloc[img_n,59],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def wearing_necklace(img_n,df):
    points = np.array([[df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],
                        df.iloc[img_n,10],df.iloc[img_n,11],[df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])],
                        [df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])]]],dtype=np.int32)
    return points

def wearing_necktie(img_n,df):
    points = np.array([[df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],
                        df.iloc[img_n,10],df.iloc[img_n,11],[df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])],
                        [df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])],
                        [df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])]]],dtype=np.int32)
    return points

def young(img_n,df):
    border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]*1.1) - df.iloc[img_n,33][1]
    if border1 < 0:
        border1 = df.iloc[img_n,24][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1]*1.3
    if border2 < 0:
        border2 = df.iloc[img_n,19][1] - (df.iloc[img_n,8][1]) - df.iloc[img_n,33][1] * 1.3

    chin = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],
    df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],df.iloc[img_n,54],df.iloc[img_n,55],
    df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],df.iloc[img_n,59],df.iloc[img_n,58]],dtype=np.int32)

    left_cheek = np.array([df.iloc[img_n,0],df.iloc[img_n,1],df.iloc[img_n,2],df.iloc[img_n,3],
        df.iloc[img_n,4],df.iloc[img_n,5],df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,39],
        df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36],df.iloc[img_n,0]],dtype=np.int32)

    right_cheek = np.array([df.iloc[img_n,16],
    df.iloc[img_n,15],df.iloc[img_n,14],df.iloc[img_n,13],df.iloc[img_n,12],df.iloc[img_n,11],
    df.iloc[img_n,54],df.iloc[img_n,35],df.iloc[img_n,42],df.iloc[img_n,47],df.iloc[img_n,46],##cheeks
    df.iloc[img_n,45],df.iloc[img_n,16]],dtype=np.int32)

    upper_lip = np.array([df.iloc[img_n,48],df.iloc[img_n,31],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],df.iloc[img_n,35],df.iloc[img_n,54],
    df.iloc[img_n,53],df.iloc[img_n,52],df.iloc[img_n,51],df.iloc[img_n,50],
    df.iloc[img_n,49],df.iloc[img_n,48]],dtype=np.int32)

    neck = np.array([df.iloc[img_n,5],df.iloc[img_n,6],df.iloc[img_n,7],df.iloc[img_n,8],df.iloc[img_n,9],df.iloc[img_n,10],df.iloc[img_n,11],
    (df.iloc[img_n,11][0],df.iloc[img_n,11][1]-(df.iloc[img_n,33][1] - df.iloc[img_n,58][1])),
    (df.iloc[img_n,10][0],df.iloc[img_n,10][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,9][0],df.iloc[img_n,9][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,8][0],df.iloc[img_n,8][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,7][0],df.iloc[img_n,7][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,6][0],df.iloc[img_n,6][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1])),
    (df.iloc[img_n,5][0],df.iloc[img_n,5][1]-(df.iloc[img_n,33][1]-df.iloc[img_n,58][1]))],dtype=np.int32)

    mouth = np.array([df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
    df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
    df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
    df.iloc[img_n,59],df.iloc[img_n,48]],dtype=np.int32)

    left_eye = np.array([df.iloc[img_n,36],df.iloc[img_n,37],df.iloc[img_n,38],
    df.iloc[img_n,39],df.iloc[img_n,40],df.iloc[img_n,41],df.iloc[img_n,36]],dtype=np.int32)

    right_eye = np.array([df.iloc[img_n,42],df.iloc[img_n,43],df.iloc[img_n,44],df.iloc[img_n,45],
    df.iloc[img_n,46],df.iloc[img_n,47],df.iloc[img_n,42]],dtype=np.int32)

    nose = np.array([[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]],df.iloc[img_n,32],
    df.iloc[img_n,33],df.iloc[img_n,34],[df.iloc[img_n,35][0]+(df.iloc[img_n,35][0]-df.iloc[img_n,34][0]),df.iloc[img_n,35][1]],
    df.iloc[img_n,42],df.iloc[img_n,27],df.iloc[img_n,39],[df.iloc[img_n,31][0]-(df.iloc[img_n,32][0]-df.iloc[img_n,31][0]),df.iloc[img_n,31][1]]],dtype=np.int32)

    left_eyebrow = np.array([df.iloc[img_n,17],df.iloc[img_n,18],df.iloc[img_n,19],
        df.iloc[img_n,20],df.iloc[img_n,21]],dtype=np.int32)

    right_eyebrow = np.array([df.iloc[img_n,22],df.iloc[img_n,23],
    df.iloc[img_n,24],df.iloc[img_n,25],df.iloc[img_n,26]],dtype=np.int32)

    top_of_head = np.array([df.iloc[img_n,24],[df.iloc[img_n,24][0],border1],
    [df.iloc[img_n,19][0],border2],df.iloc[img_n,19]],dtype=np.int32)

    left_ear = np.array([[df.iloc[img_n,0][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,0][0]),df.iloc[img_n,0][1]],
    [df.iloc[img_n,1][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,1][0]),df.iloc[img_n,1][1]],
    [df.iloc[img_n,2][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,2][0]),df.iloc[img_n,2][1]],
    [df.iloc[img_n,3][0]-(df.iloc[img_n,17][0]-df.iloc[img_n,3][0]),df.iloc[img_n,3][1]],
    df.iloc[img_n,3],df.iloc[img_n,2],df.iloc[img_n,1],df.iloc[img_n,0]],dtype=np.int32)

    right_ear = np.array([[df.iloc[img_n,16][0]+(df.iloc[img_n,16][0]-df.iloc[img_n,26][0]),df.iloc[img_n,16][1]],
    [df.iloc[img_n,15][0]+(df.iloc[img_n,15][0]-df.iloc[img_n,26][0]),df.iloc[img_n,15][1]],
    [df.iloc[img_n,14][0]+(df.iloc[img_n,14][0]-df.iloc[img_n,26][0]),df.iloc[img_n,14][1]],
    [df.iloc[img_n,13][0]+(df.iloc[img_n,13][0]-df.iloc[img_n,26][0]),df.iloc[img_n,13][1]],
    df.iloc[img_n,13],df.iloc[img_n,14],df.iloc[img_n,15],df.iloc[img_n,16]],dtype=np.int32)


    points = [chin, left_cheek, right_cheek, upper_lip, neck, mouth, left_eye, right_eye, nose, left_eyebrow, right_eyebrow, top_of_head, left_ear, right_ear]
    return points

fn_dict = {
    '5_o_Clock_Shadow': five_oclock_shadow,
    'Arched_Eyebrows': arched_eyebrows,
    'Attractive': attractive,
    'Bags_Under_Eyes': bags_under_eyes,
    'Bald': bald,
    'Bangs': bangs,
    'Big_Lips': big_lips,
    'Big_Nose': big_nose,
    'Black_Hair': black_hair,
    'Blond_Hair': blond_hair,
    'Blurry': blurry,
    'Brown_Hair': brown_hair,
    'Bushy_Eyebrows': bushy_eyebrows,
    'Chubby': chubby,
    'Double_Chin': double_chin,
    'Eyeglasses': eyeglasses,
    'Goatee': goatee,
    'Gray_Hair': gray_hair,
    'Heavy_Makeup': heavy_makeup,
    'High_Cheekbones': high_cheekbones,
    'Male': male,
    'Mouth_Slightly_Open': mouth_slightly_open,
    'Mustache': mustache,
    'Narrow_Eyes': narrow_eyes,
    'No_Beard': no_beard,
    'Oval_Face': oval_face,
    'Pale_Skin': pale_skin,
    'Pointy_Nose': pointy_nose,
    'Receding_Hairline': receding_hairline,
    'Rosy_Cheeks': rosy_cheeks,
    'Sideburns': sideburns,
    'Smiling': smiling,
    'Straight_Hair': straight_hair,
    'Wavy_Hair': wavy_hair,
    'Wearing_Earrings': wearing_earrings,
    'Wearing_Hat': wearing_hat,
    'Wearing_Lipstick': wearing_lipstick,
    'Wearing_Necklace': wearing_necklace,
    'Wearing_Necktie': wearing_necktie,
    'Young': young
}

def binarize(arr):
    bin = np.zeros(arr.shape)
    for x,row in enumerate(arr):
        for y, cell in enumerate(row):
            if(cell == 255):
                bin[x,y] = 1
    return bin

def generate_masks(img,name,index,df):
    for fn in fn_dict.keys():
        points = fn_dict[fn](index,df)
        #print(img.shape, name)
        img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bin.fill(0)
        if points is not None:
            for point in points:
                point = np.expand_dims(point,axis=0)
                cv2.fillPoly(img=img_bin,pts=point,color=255,lineType=cv2.LINE_AA)
        img_bin = crop_openface(img_bin,df.iloc[index],cropsize,name)
        img_bin = (img_bin > 0).astype(np.uint8) * 255
        file = binarize(img_bin)
        np.save("/home/guillermodelvalle/OpenFace_masks/npy/"+name+"_"+fn,file)
        yield img_bin, fn

def process_images(df,features,dir,out,id):
    labels = features.columns.values
    rows = df.index.values
    file_df = pd.DataFrame(columns=labels,index=rows)
    for i in range(len(df)):
        name = df.index[i]
        #print(name)
        img = cv2.imread(str(dir+'/'+name))
        file_df.at[name,"image_id"] = name
        for mask,fn in generate_masks(img,name[:-4],i,df):
            cv2.imwrite(os.path.join("/home/guillermodelvalle/OpenFace_masks/jpg/"+name[:-4]+'_'+fn+'.jpg'), mask)
            file_df.at[name,fn] = name[:-4]+'_'+fn+'.jpg'
    #print(file_df)
    file_df.to_csv(id+"final.csv")
    return



dict = use_bbox('list_bbox_celeba.csv')
#print(dict['000001.jpg'])
csv_file = 'test.csv'
path = '/home/guillermodelvalle/img_celeba'
#OpenFaceBashCommand = '/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -fdir '+path+' -out_dir ../OpenFace_landmarks/'
#print(OpenFaceBashCommand)
#process_directory(path, csv_file, dict)
df = process_directory_openface(path, csv_file, dict)
#print(df)
#print(df.iloc[0,0])
features = create_new_csv(df,'list_attr_celeba.csv')
#print(features)
print(df)
process_images(df,features,path,'/home/guillermodelvalle/OpenFace_detected/','1')
#print("Percentage of found:", found/(not_found+found))
opencvdf = process_directory('/home/guillermodelvalle/OpenFace_not_detected', csv_file, dict)
opencvfeatures = create_new_csv(opencvdf,'list_attr_celeba.csv')
print(opencvdf)
process_images(opencvdf,opencvfeatures,path,'/home/guillermodelvalle/detected_opencv','2')
end = time.time()
print("Time taken:", end - start)
print(missed_count)
