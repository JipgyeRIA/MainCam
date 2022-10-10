import cv2
from collections import deque
import face_recognition
from multiprocessing import Process,Queue
import time

def face_find_func(img_process_queue, result_process_queue):
    while True:
        if img_process_queue.qsize() != 0:
            start = time.time()
            img = img_process_queue.get()
            face_loc = face_recognition.face_locations(img)
            result_process_queue.put(face_loc)
            #print("time :", time.time() - start)



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
        

def img_show_func( img_process_queue, result_process_queue, video = None):



    move_face_len = deque([])
    queue_size = 10
    group_threshold = 0.5
    group_previous_flag = False
    group_flag = False

    scaling_factor, restoring_factor, margin = 0.5,2,20

    if video ==None:
        pass
    else:
        vid = cv2.VideoCapture(video)
        if vid.isOpened() == False:
            exit()

    face_loc = []
    face_detector_lock = False
    while True:
        ret,frame = vid.read()
        if ret:
            resized_frame = cv2.resize(frame, dsize=(0, 0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            if face_detector_lock == False:
                face_detector_lock = True
                img_process_queue.put(resized_frame)

            while result_process_queue.qsize() != 0:
                face_detector_lock = False
                face_loc = result_process_queue.get()
            #top, right, bottom, left
            if len(face_loc) != 0:
                for (top,right,bottom,left) in face_loc:
                    top,right,bottom,left = [ val * restoring_factor for val in  (top,right,bottom,left)]
                    top,right,bottom,left = top-margin,right+margin,bottom+margin,left-margin
                    cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))
            

            group_previous_flag = group_flag
            group_flag = group_checker(move_face_len,len(face_loc))
            #print(group_flag)
            if group_previous_flag == False and group_flag == True:
                print("group_start")
            
            if group_previous_flag == True and group_flag == False:
                print("group_end")

            cv2.imshow("test",frame)
            cv2.waitKey(60)
        else:
            break

if __name__ =='__main__':
    # Code without daemon
    img_process_queue = Queue()
    result_process_queue = Queue()

    img_proc = Process(target=img_show_func ,args=(img_process_queue,result_process_queue,"./test_img/test_video_1.mp4",),daemon=True)
    face_proc = Process(target=face_find_func ,args=(img_process_queue,result_process_queue,),daemon=True)

    img_proc.start()
    face_proc.start()

    img_proc.join()
    face_proc.join()




