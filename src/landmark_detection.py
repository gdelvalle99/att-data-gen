import numpy as np
import cv2
import pandas as pd
import os
import dlib
import csv
import time
from natsort import natsorted
import subprocess
import point
import shutil

start = time.time()

found = 0
not_found = 0
#detector = dlib.get_frontal_face_detector()

#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#imported from face_utils

def crop_openface(img,bbox):
    #bbox[24]
    #print(img)
    dimensions = img.shape

    top = (bbox[24][1] - (bbox[8][1]*1.1))
    if(top < 0):
        top = 0
    left = bbox[0][0] - (bbox[36][0]-bbox[0][0])
    if(left < 0):
        left = 0
    right = bbox[16][0] + (bbox[16][0]-bbox[45][0])
    if(right > img.shape[1]):
        right = img.shape[1]-1

    bottom = bbox[8][1] - (bbox[33][1] - bbox[8][1])
    if(bottom > img.shape[0]):
        bottom = img.shape[0]-1
    #right = bbox[17]
    #bottom = bbox[9]
    print(top,left,right,bottom)
    crop = img[int(top):int(bottom), int(left):int(right)]
    print(crop)
    crop = cv2.resize(crop,(178,218))
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
def extract_landmarks_opencv(name,shape, df):

    list = []
    for i in shape:
        list.append(i)
    #appends x and y coordinates as a single tuple
    df_length = len(df)
    df.loc[df_length]= list
    df.index = df.index[:-1].tolist() + [name]


    return df

def extract_landmarks_openface(name,dir,df,file_name,dict,out):
    entry = dir + name[:-4] + ".csv"
    if(os.path.isfile(entry) is False):
        if(os.path.isdir(file_name+name) is True):
            return df
        filename = file_name+name
        #print(file_name)
        shutil.copy(filename,file_name+"/OpenFace_not_detected")
        return df
    else:
        #print(file_name+name)
        filename = file_name+'/' + name
        #print(filename)
        #print(file_name)
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
        img = crop_openface(img,list)
        #print(file_name+'OpenFace_detected/'+name)
        cv2.imwrite(out+"/"+name,img)
        print(out+"/"+name)
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
      #  print(rects)
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
        for i in range(of_landmarks.index):
            coords = np.array(((int(round(of_landmarks.iloc[i][0]))), (int(round(of_landmarks.iloc[i][68])))))
            ##dlib_bbox = np.array((x,y))
            #print(celebA_bbox - dlib_bbox)
            if np.linalg.norm(celebA_bbox - coords) < dist:
                closest_bbox = i
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
    if not os.path.exists("OpenFace_detected"):
        os.mkdir("OpenFace_detected")
    if not os.path.exists("OpenFace_not_detected"):
        os.mkdir("OpenFace_not_detected")
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
    entries = natsorted(os.listdir(dir))
    for entry in entries:
        df = extract_landmarks_openface(entry,'../OpenFace_landmarks/',df,dir,dict[entry],"OpenFace_detected")
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



def Big_Lips(img_n,df):
    points = np.array([[df.iloc[img_n,48],df.iloc[img_n,49],df.iloc[img_n,50],
        df.iloc[img_n,51],df.iloc[img_n,52],df.iloc[img_n,53],df.iloc[img_n,54],
        df.iloc[img_n,55],df.iloc[img_n,56],df.iloc[img_n,57],df.iloc[img_n,58],
        df.iloc[img_n,59],df.iloc[img_n,48]]],dtype=np.int32)
    return points

def generate_masks(img,index,df):
    points = Big_Lips(index,df)
    #print(img)
    img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin.fill(0)
    cv2.fillPoly(img=img_bin,pts=points,color=255,lineType=cv2.LINE_AA)
    arr = np.array(img_bin)
    np.save("npy_"+str(index),arr)
    return img_bin

def process_images(df,features,dir):
    for i in range(len(df)):
        name = df.index[i]
        print(dir+'/'+name)
        img = cv2.imread(str(dir+'/'+name))
        mask = generate_masks(img,i,df)
        cv2.imwrite(os.path.join(dir+'/'+'mask_'+str(i)+'.jpg'), mask)
    return

dict = use_bbox('list_bbox_celeba.csv')
#print(dict['000001.jpg'])
csv_file = 'test.csv'
path = '/home/guillermodelvalle/att-data-gen/src/images'
#OpenFaceBashCommand = '/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -fdir '+path+' -out_dir ../OpenFace_landmarks/'
#print(OpenFaceBashCommand)
#process_directory(path, csv_file, dict)
df = process_directory_openface(path, csv_file, dict)
#print(df)
#print(df.iloc[0,0])
features = create_new_csv(df,'list_attr_celeba.csv')
#print(features)
process_images(df,features,'OpenFace_detected')
#print("Percentage of found:", found/(not_found+found))
#end = time.time()
#print("Time taken:", end - start)
