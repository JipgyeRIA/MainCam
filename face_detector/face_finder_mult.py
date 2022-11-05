import cv2
from collections import deque
import face_recognition
from multiprocessing import Process,Queue
import time
import numpy as np
from datetime import datetime
from face_gender_age_mult import gender_age_func
import os

def face_find_func(img_process_queue, result_process_queue, scaling_factor = 0.5,restoring_factor = 2):

    naive_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    margin = 30
    while True:
        if img_process_queue.qsize() != 0:
            start = time.time()
            img = img_process_queue.get()
            if not isinstance(img, np.ndarray):
                break

            H,W,_ = img.shape
            resized_img = cv2.resize(img, dsize=(0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            face_loc = face_recognition.face_locations(resized_img)
            result_loc = []

            for loc in face_loc:
                top,right, bottom,left = [val * restoring_factor for val in loc]
                im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)

                gray_img = cv2.cvtColor(img[im_t:im_b,im_l:im_r] , cv2.COLOR_BGR2GRAY)
                naive_face = naive_classifier.detectMultiScale(gray_img)
                conf = 0
                if len(naive_face) ==1:
                    conf = 1
                result_loc.append([top,right, bottom,left,conf])


            result_process_queue.put([img,result_loc])
            #print("time :", time.time() - start)
    result_process_queue.put(-1)
    print("finder End")


def group_checker(group_queue,face_len,queue_size = 10,threshold = 0.8):
    #self.move_face_len.append(face_len)
    group_queue.append( 1 if face_len > 0 else 0)
    if len(group_queue) > queue_size:
        group_queue.popleft()

        avg = sum(group_queue) / len(group_queue)

        if avg > threshold:
            return True
        else:
            return False
    else:
        return False
        

def img_show_func( inst_queue, result_queue, video = None):
    finder_inst_queue = inst_queue[0]
    finder_result_queue = result_queue[0]

    sending_inst_queue = inst_queue[1]
    sending_result_queue =result_queue[1]


    move_face_len = deque([])
    group_previous_flag = False
    group_flag = False

    group_key = ""
    group_img = None
    group_conf_max = -1
    group_len = -1
    group_loc = []

    margin = 30

    if video ==None:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video)
        if vid.isOpened() == False:
            exit()

    face_loc = []
    face_detector_lock = False
    while True:
        ret,frame = vid.read()
        if ret:
            H,W,_ = frame.shape
            

            ################ face Detection ############
            if face_detector_lock == False:
                face_detector_lock = True
                finder_inst_queue.put(frame.copy())

            while finder_result_queue.qsize() != 0:
                face_detector_lock = False
                processed_img, face_loc = finder_result_queue.get()
            #top, right, bottom, left

            if len(face_loc) != 0:
                conf_mean = 0
                for (top,right,bottom,left,conf) in face_loc:
                    top,right,bottom,left = top-margin,right+margin,bottom+margin,left-margin
                    im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)
                    cv2.rectangle(frame, [im_l,im_t],[im_r,im_b],(0,255,0))
                    conf_mean +=conf
                conf_mean /= len(face_loc)
    
                if group_flag == True:
                    if len(face_loc) > group_len:
                        group_img = processed_img
                        group_len = len(face_loc)
                        group_conf_max = conf_mean
                        group_loc = face_loc
                    elif len(face_loc) == group_len and conf_mean > group_conf_max:
                        group_img = processed_img
                        group_conf_max = conf_mean
                        group_loc = face_loc


            group_previous_flag = group_flag
            group_flag = group_checker(move_face_len,len(face_loc))
            #print(group_flag)


            # Group Checking & Find Best Group IMG
            ###################################################


            while sending_result_queue.qsize() != 0:
                sending_ret = sending_result_queue.get()
                if sending_ret != 1:
                    # Error
                    print(f"{sending_ret} ERROR")
        

            if group_previous_flag == False and group_flag == True:
                print("group_start")
                group_key = datetime.now().strftime('%Y%m%d%H%M%S')
                group_img = None
                group_conf_max = -1
                group_len = -1
                group_loc = []

                # initialize_group
                
                # Group Start Function

            
            if group_previous_flag == True and group_flag == False:
                #print(group_img,group_loc,group_key)

                sending_inst_queue.put([group_img,group_loc,group_key])

                # for (top,right,bottom,left,conf) in group_loc:
                #     top,right,bottom,left = top-margin,right+margin,bottom+margin,left-margin
                #     im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)
                #     cv2.rectangle(group_img, [im_l,im_t],[im_r,im_b],(0,255,0))
                # cv2.imshow(group_key,group_img)
                group_key = ""
                group_img = None
                group_conf_max = -1
                group_len = -1
                group_loc = []
                print("group_end")

                # Group End Function

            cv2.imshow("frame multiprocess",frame)
            cv2.waitKey(33)
        else:
            # For End Request
            finder_inst_queue.put(-1)
            sending_inst_queue.put(-1)

            end_flag = [False,False]
            while True:
                if finder_result_queue.qsize() != 0:
                    ret = finder_result_queue.get()
                    if ret == -1:
                        end_flag[0] = True
                if sending_result_queue.qsize() !=0:
                    ret = sending_result_queue.get()
                    if ret == -1:
                        end_flag[1] = True
                
                if end_flag[0] and end_flag[1]:
                    break
            break
    print("Proc End")


if __name__ =='__main__':
    # Code without daemon
    finder_inst_queue = Queue()
    finder_result_queue = Queue()

    sending_inst_queue = Queue()
    sending_result_queue = Queue()




    img_proc = Process(target=img_show_func ,args=([finder_inst_queue, sending_inst_queue],
                                                [finder_result_queue, sending_result_queue],
                                                "./test_img/test_video_2.mp4",),
                                                daemon=True)
        
    face_proc = Process(target=face_find_func ,args=(finder_inst_queue,finder_result_queue,),daemon=True)

    send_proc = Process(target=gender_age_func ,args=(sending_inst_queue,sending_result_queue,),daemon=True)

    



    img_proc.start()
    face_proc.start()
    send_proc.start()

    send_proc.join()
    face_proc.join()
    img_proc.join()
    


    # img_proc.close()
    # face_proc.close()

    print("Program End")




