import cv2
import numpy as np
import dlib
from math import hypot
import time
import pyautogui

def midpoint(middleOfEyeLandMark1,middleOfEyeLandMark2):
    return int((middleOfEyeLandMark1.x + middleOfEyeLandMark2.x)/2), int((middleOfEyeLandMark1.y,middleOfEyeLandMark2.y))

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def main():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    font = cv2.FONT_HERSHEY_PLAIN
    while True:

        _, frame = cap.read()
        new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            '''
            tl = [landmarks.part(36).x,landmarks.part(37).y-10]
            tr = [landmarks.part(39).x,landmarks.part(38).y-10]
            bl = [landmarks.part(36).x,landmarks.part(41).y+10]
            br = [landmarks.part(39).x,landmarks.part(40).y+10]
            pts =np.array([tl,tr,bl,br],np.int32)
            rect = order_points(pts)
            warped = four_point_transform(frame,pts)
            left_eye_region = warped
            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            left_eye = cv2.bitwise_and(gray, gray, mask=mask)
            #cv2.polylines(mask, [left_eye_region], True, 255, 2)
            '''
            #cv2.fillPoly(mask, [left_eye_region], 255)
            '''\
            left_eye = cv2.bitwise_and(gray, gray, mask=mask)
            theLeftMostSideEyeX = np.min(left_eye_region[:,0])
            theRightMostSideEyeX = np.max(left_eye_region[:,0])
            theLeftMostSideEyeY = np.min(left_eye_region[:,1])
            theRightMostSideEyeY = np.max(left_eye_region[:,1])
            '''
            #eye = left_eye[theLeftMostSideEyeY:theRightMostSideEyeY,theLeftMostSideEyeX:theRightMostSideEyeX]
            #_, threshold_eye = cv2.threshold(warped,50,255,cv2.THRESH_BINARY)

            #print(landmarks.part(37).y-10-(landmarks.part(41).y+10))
            #print(landmarks.part(36).x-(landmarks.part(39).x))
            #cv2.rectangle(frame,tl,br,(0,0,255),2)

            left_eye_region = np.array([(landmarks.part(36).x,landmarks.part(36).y),
                                        (landmarks.part(37).x,landmarks.part(37).y),
                                        (landmarks.part(38).x,landmarks.part(38).y),
                                        (landmarks.part(39).x,landmarks.part(39).y),
                                        (landmarks.part(40).x,landmarks.part(40).y),
                                        (landmarks.part(41).x,landmarks.part(41).y)],np.int32)

            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [left_eye_region], True, 255, 2)
            cv2.fillPoly(mask, [left_eye_region], 255)
            left_eye = cv2.bitwise_and(gray, gray, mask=mask)

            theLeftMostSideEyeX = np.min(left_eye_region[:,0])
            theRightMostSideEyeX = np.max(left_eye_region[:,0])
            theLeftMostSideEyeY = np.min(left_eye_region[:,1])
            theRightMostSideEyeY = np.max(left_eye_region[:,1])


            eye = left_eye[theLeftMostSideEyeY:theRightMostSideEyeY,theLeftMostSideEyeX:theRightMostSideEyeX]
            #eyeGrayScaled= cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
            _, threshold_eye = cv2.threshold(eye,50,255,cv2.THRESH_BINARY)

            heightOfEye, WidthOfEye = threshold_eye.shape
            left_side_threshold = threshold_eye[0:heightOfEye, 0:int(WidthOfEye/2)]
            left_side_white = cv2.countNonZero(left_side_threshold)

            right_side_threshold = threshold_eye[0:heightOfEye, int(WidthOfEye/2):width]
            right_side_white = cv2.countNonZero(right_side_threshold)

            try:
                gaze_ratio = left_side_white/right_side_white
            except:
                passa

            if gaze_ratio < 0.6:
                print("right")
                pyautogui.scroll(75)
            if gaze_ratio > 1.5:
                print('left')
                pyautogui.scroll(-50)
            else:
                print('center')

            #print(gaze_ratio)
            #time.sleep(1)
            #row_sum=np.sum(threshold_eye, axis=1)
            '''
            threshold_eye=threshold_eye/255
            print(threshold_eye)
            print(threshold_eye.shape)
            i=threshold_eye.sum(axis=1).argmax()
            j=threshold_eye.sum(axis=0).argmax()
            pyautogui.moveTo(int(1920/threshold_eye.shape[0]*i),int(1200/threshold_eye.shape[1]*j),duration=2)'''

            #threshold_eye = cv2.resize(threshold_eye,None,fx=5,fy=5)
            #eye = cv2.resize(eye,None,fx=5,fy=5)
            #cv2.imshow("EYE",eye)
            #cv2.imshow("Threshold",threshold_eye)

           # cv2.imshow("EYE",eye)
        #cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord('a'):
            print("pressed a")
            break

    cap.release()
    cv2.destroyAllWindows()

main()
