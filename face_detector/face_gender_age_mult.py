import math
import pstats
import face_recognition
import cv2
from sklearn import model_selection
import time
from datetime import datetime
import os
import requests
import json
from deepface import DeepFace
from torch import align_tensors

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

def gender_model_classifying(model,img, model_mean = (78.4263377603, 87.7689143744, 114.895847746)):
    blob = cv2.dnn.blobFromImage(img, 1, (227, 227),model_mean, swapRB=False)
    #   gender_list = ['Male', 'Female']
    model.setInput(blob)
    answers = model.forward()
    #print(answers)
    answer = answers.argmax()

    return answer

def age_model_classifying(model,img, model_mean = (78.4263377603, 87.7689143744, 114.895847746),age_label = True):
    blob = cv2.dnn.blobFromImage(img, 1, (227, 227),model_mean, swapRB=False)
    #   age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)','(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']
    #   age_compute = [ { 0, 1 }, {2,3,4}, {5,6}, {7,8}]
    age_list = [1, 5, 10, 18, 29, 41, 51, 80 ]
    model.setInput(blob)
    answers = model.forward()
    answers[0][0] = answers[0][0]+answers[0][1]
    answers[0][1] = answers[0][2]+answers[0][3]+answers[0][4]
    answers[0][2] = answers[0][5]+answers[0][6]
    answers[0][4] = answers[0][7]
    answers = answers[:,:5]
    answer = answers.argmax()
    #answer = age_list[answers.argmax()]

    return answer

def send_ssh(group_dic, ip_endpoint = "https://jipgyeria.herokuapp.com/face/group"):

    #headers = {"content-type": "application/json", "Authorization": "<auth-key>" }
    val = {
    "group":group_dic
    }
    val = json.dumps(val)
    print(f"SENDING : {val} \n")

    requests.post(ip_endpoint, data=val, headers= {"content-type": "application/json"})
    return 1

def gender_age_func(img_process_queue, result_process_queue):

    def age_labeler(input):
        if input < 6:
            return 0
        elif input < 38:
            return 1
        elif input < 60:
            return 2
        else:
            return 3


    # caffe, inter, deep
    # 어떤 걸로 우선 순위를 할지 고민, inter의 경우 deeplearning의 비중
    age_policy = 0.7
    gender_policy = "deep"



    model_weight_path = "./weight_file"
    age_model  = os.path.join(model_weight_path,"deploy_age.prototxt")
    age_weight = os.path.join(model_weight_path,"age_net.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe(age_model,age_weight)

    gender_model = os.path.join(model_weight_path,"deploy_gender.prototxt")
    gender_weight = os.path.join(model_weight_path,"gender_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(gender_model,gender_weight)

    margin = 20

    while True:
        if img_process_queue.qsize() != 0:
            val = img_process_queue.get()
            if val == -1:
                break
                
            img,face_loc,group_key = val
            face_embedding = face_recognition.face_encodings(img,face_loc)
            H,W,_ = img.shape

            group_mem_dic={}
            
            for idx, ((top,right,bottom,left,_),emb) in enumerate(zip(face_loc,face_embedding)):
                im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)
                age = age_model_classifying(age_net,img[im_t:im_b,im_l:im_r])
                gender = gender_model_classifying(gender_net,img[im_t:im_b,im_l:im_r])

                response=DeepFace.analyze(img[im_t:im_b,im_l:im_r],
                                    actions=["gender","age"],
                                    enforce_detection=False,
                                    prog_bar= False,
                                    detector_backend="dlib")

                #print("caffe",gender, age)
                if age_policy == "caffe":
                    pass
                elif age_policy == "Deep":
                    age = age_labeler(response["age"])
                else:
                    val = age_labeler(response["age"])
                    age = age * (1-age_policy) + val * age_policy
                    #print(age)
                    age = int(round(age))
                
                if gender_policy == "caffe":
                    pass
                else:
                    gender = 0 if response["gender"] == "Man" else 1

                #print(gender,age)


                group_mem_dic[f"p{idx}"]={
                    "emb": embedding_2_str(emb),
                    "age": int(age),
                    "gender": int(gender),
                    "group":group_key
                }

            send_ssh(group_mem_dic)

            result_process_queue.put(1)

            

            #print("time :", time.time() - start)
    result_process_queue.put(-1)
    print("Sender End")