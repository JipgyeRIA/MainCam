import math
import face_recognition
import cv2
from sklearn import model_selection
import time
from datetime import datetime
import os

def embedding_2_str(face_embedding, precision = 4, min_val = 1.0):
    # precision mean 0.000000 -> 6  아래 자리수 표현 즉 6일때 7칸 필요
    face_str = ""
    for val in face_embedding:
        val += min_val
        val = str(val)
        val = val.replace(".","",1)
        val += "0" * precision
        val = val[:precision+1]
        face_str += val

    return face_str

def str_2_embedding(face_str,precision = 4, embedding_size = 128):

    face_embedding = []
    for idx in range(embedding_size):
        val = face_str[idx * (precision+1) : (idx+1) * (precision+1)]
        val = val[0] +"." + val[1:]
        val = float(val)
        #print(val)
        #val -= min_val
        face_embedding.append(val)

    return face_embedding


def l2_norm(x,y):
    norm_val = 0
    for a,b in zip(x,y):
        norm_val += (a-b)*(a-b)
    
    norm_val = math.sqrt(norm_val)
    return norm_val

def model_classifying(model,img, model_mean = (78.4263377603, 87.7689143744, 114.895847746)):
    blob = cv2.dnn.blobFromImage(img, 1, (227, 227),model_mean, swapRB=False)
    
    model.setInput(blob)
    answers = model.forward()
    answer = answers.argmax()

    return answer

def send_ssh(embedding,age,gender,group_key):
    emb_str = embedding_2_str(embedding)
    print(f"SENDING : {emb_str} , {age}, {gender} in {group_key} \n")
    return 1

def gender_age_func(img_process_queue, result_process_queue):
    

    ### Age and Gender label ###
    #
    #   age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)','(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']
    #   gender_list = ['Male', 'Female']
    #
    ############################
    model_weight_path = "./weight_file"
    age_model  = os.path.join(model_weight_path,"deploy_age.prototxt")
    age_weight = os.path.join(model_weight_path,"age_net.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe(age_model,age_weight)

    gender_model = os.path.join(model_weight_path,"deploy_gender.prototxt")
    gender_weight = os.path.join(model_weight_path,"gender_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(gender_model,gender_weight)

    margin = 30

    print("hello")
    while True:
        if img_process_queue.qsize() != 0:
            val = img_process_queue.get()
            if val == -1:
                break
                
            img,face_loc,group_key = val
            face_embedding = face_recognition.face_encodings(img,face_loc)
            H,W,_ = img.shape

            response = 0
            for (top,right,bottom,left,_),emb in zip(face_loc,face_embedding):
                im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)
                age = model_classifying(age_net,img[im_t:im_b,im_l:im_r])
                gender = model_classifying(gender_net,img[im_t:im_b,im_l:im_r])
                
                response += send_ssh(emb,age,gender,group_key)

            if response == len(face_loc):
                result_process_queue.put(1)
            else:
                result_process_queue.put(group_key)
            

            #print("time :", time.time() - start)
    result_process_queue.put(-1)
    print("Sender End")