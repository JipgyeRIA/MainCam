from deepface import DeepFace
import cv2
import face_recognition
from multiprocessing import Process,Queue
import numpy as np
from collections import deque

def gender_age_func(img_process_queue, result_process_queue):
    while True:
        img = None
        while img_process_queue.qsize() != 0:
            img = img_process_queue.get()
            if not isinstance(img, np.ndarray):
                result_process_queue.put(-1)

                return -1
        if isinstance(img, np.ndarray):
            response=DeepFace.analyze(img,
                        actions=["gender","age"],
                        enforce_detection=False,
                        prog_bar= False,
                        detector_backend="dlib")


            result_process_queue.put([img,response])


    result_process_queue.put(-1)

    

def face_find_func(img_process_queue, result_process_queue, scaling_factor = 0.5,restoring_factor = 2):

    naive_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    margin = 30
    while True:
        if img_process_queue.qsize() != 0:
            img = img_process_queue.get()
            if not isinstance(img, np.ndarray):
                break

            H,W,_ = img.shape
            result_loc = []

            resized_img = cv2.resize(img, dsize=(0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            face_loc = face_recognition.face_locations(resized_img)

            if len(face_loc) != 0 :
                maximum_face_loc = []
                max_face_size = -1
                max_face_idx = -1
                for i,loc in enumerate(face_loc):
                    top,right, bottom,left = [val * restoring_factor for val in loc]
                    tmp_size = (bottom - top) * (right - left)
                    if max_face_size < tmp_size:
                        maximum_face_loc = [top,right, bottom,left]
                        max_face_size = tmp_size
                        max_face_idx = i

                    result_loc.append([top,right, bottom,left,0])
            


                # find Largest Image

                top,right, bottom,left = maximum_face_loc
                im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)

                naive_face = naive_classifier.detectMultiScale(\
                                            cv2.cvtColor(img[im_t:im_b,im_l:im_r] , cv2.COLOR_BGR2GRAY) )
                
                if len(naive_face) ==1:
                    naive_size = naive_face[0][2] * naive_face[0][3]
                    if naive_size > max_face_size * 0.5 :
                        result_loc[max_face_idx][-1] = 1

            result_process_queue.put([img,result_loc])
            #print("time :", time.time() - start)
    result_process_queue.put(-1)
    print("finder End")



def img_show_func( inst_queue, result_queue, margin = 30, show_len = 2 ):
    finder_inst_queue = inst_queue[0]
    finder_result_queue = result_queue[0]

    age_gender_inst_queue = inst_queue[1]
    age_gender_result_queue = result_queue[1]

    vid = cv2.VideoCapture(0)
    if vid.isOpened() == False:
        exit()

    face_detector_lock = False
    face_loc = []
    #gender_making_face = [np.zeros((120,120,3),dtype=np.uint8) for _ in range(4)]

    gender_making_face = deque([])

    while True:
        ret,frame = vid.read()
        if ret:
            H,W,_ = frame.shape

            if H % show_len != 0:
                break
            
            ################ face Detection ############
            if face_detector_lock == False:
                face_detector_lock = True
                finder_inst_queue.put(frame.copy())

            
            while finder_result_queue.qsize() != 0:
                face_detector_lock = False
                processed_img, face_loc = finder_result_queue.get()
            #top, right, bottom, left

            if len(face_loc) != 0:
                print(face_loc)
                for (top,right,bottom,left,conf) in face_loc:
                    top,right,bottom,left = top-margin,right+margin,bottom+margin,left-margin
                    im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)
                    cv2.rectangle(frame, [im_l,im_t],[im_r,im_b],(0,255,0))
                    #cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))

                    if conf == 1:
                        face_front_img = processed_img[im_t:im_b,im_l:im_r] 
                        age_gender_inst_queue.put(face_front_img)



            while age_gender_result_queue.qsize() != 0:
                gender_age_img, age_gender  = age_gender_result_queue.get()
                gender_making_face.append([gender_age_img,age_gender])
                if len(gender_making_face) > show_len:
                    gender_making_face.popleft()


            add_on_img = []
            for i in range(show_len):
                
                if i < len(gender_making_face):
                    tmp,age_gender = gender_making_face[i]
                    tmp = cv2.resize(tmp,(H//show_len,H//show_len))

                    age, gender = age_gender["age"],age_gender["gender"]
                    cv2.putText(tmp, f"Age:{age},gender:{gender} ", (20, 20), 0 , 0.5,(255,255,255), 1, cv2.LINE_AA) 
                else:
                    tmp = np.zeros((H//show_len,H//show_len,3),dtype=np.uint8)
                cv2.rectangle(tmp, [0,0],[H//show_len,H//show_len],(255,255,255))
                add_on_img.append(tmp)
            
            h_img = cv2.vconcat([tmp for tmp in reversed(add_on_img)])
            frame = cv2.hconcat([frame,h_img])
            cv2.imshow("frame multiprocess",frame)
            cv2.waitKey(33)


        else:
            # For End Request
            finder_inst_queue.put(-1)
            age_gender_inst_queue.put(-1)

            end_flag = [False,False]
            while True:
                if finder_result_queue.qsize() != 0:
                    ret = finder_result_queue.get()
                    if ret == -1:
                        end_flag[0] = True
                if age_gender_result_queue.qsize() !=0:
                    ret = age_gender_result_queue.get()
                    if ret == -1:
                        end_flag[1] = True
                
                if end_flag[0] and end_flag[1]:
                    break
            break
    print("Proc End")





if __name__ =='__main__':


    finder_inst_queue = Queue()
    finder_result_queue = Queue()

    age_gender_inst_queue = Queue()
    age_gender_result_queue = Queue()


    img_proc = Process(target=img_show_func ,args=([finder_inst_queue,age_gender_inst_queue],
                                                [finder_result_queue,age_gender_result_queue],),
                                                daemon=True)
    face_proc = Process(target=face_find_func ,args=(finder_inst_queue,finder_result_queue,),daemon=True)
    age_gender_proc = Process(target=gender_age_func ,args=(age_gender_inst_queue,age_gender_result_queue,),daemon=True)
    
    img_proc.start()
    face_proc.start()
    age_gender_proc.start()

    age_gender_proc.join()
    face_proc.join()
    img_proc.join()