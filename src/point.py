import cv2
import dlib

detector = dlib.get_frontal_face_detector()
img = cv2.imread("004151.jpg")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(img_gray,1)
for (i,rect) in enumerate(rects):
    shape = predictor(img_gray, rect)
    shape = shape_to_np(shape)
    (x,y,w,h) = rect_to_bb(rect)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0))

    for(x,y) in shape:
        cv2.circle(img, (x,y), 1, (0,0,255),-1)

cv2.imshow("input",img)
cv2.waitKey(0)
