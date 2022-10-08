import numpy as np
import cv2
from collections import deque
import face_recognition

class face_finder:
    def __init__(self,video = None) -> None:
        if video ==None:
            pass
        else:
            self.cap = cv2.VideoCapture("test_video_1.mp4")
            if self.cap.isOpened() == False:
                exit()

        self.move_face_len = deque([])
        self.queue_size = 10
        self.group_threshold = 0.5

        self.group_in_flag = False
        #self.move_img = deque([])


    def get_img_with_face(self):
        ret,frame = self.cap.read()
        if ret:
            face_loc = face_recognition.face_locations(frame)
            print("Number of faces detected: " + str(len(face_loc)))

            #top, right, bottom, left
            # for (top,right,bottom,left) in face_loc:
            #     cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))            

        return face_loc,frame


    def group_checker(self,img,face_len):
        #self.move_face_len.append(face_len)
        self.move_face_len.append( 1 if face_len > 0 else 0)
        if len(self.move_face_len) > self.queue_size:
            self.move_face_len.popleft()

            avg = sum(self.move_face_len) / len(self.move_face_len)

            if avg > self.group_threshold:
                self.group_in_flag = True
            else:
                self.group_in_flag = False
                
        


while True:
    ret,frame = cap.read()
    if ret:
        face_loc = face_recognition.face_locations(frame)
        print("Number of faces detected: " + str(len(face_loc)))

        #top, right, bottom, left
        for (top,right,bottom,left) in face_loc:
            cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))
        cv2.imshow("test",frame)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
