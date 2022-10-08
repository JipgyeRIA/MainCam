from operator import mod
from cv2 import EMD
from deepface import DeepFace
from deepface.commons import functions
import numpy as np
import cv2
from retinaface import RetinaFace
import face_recognition
import os
import math

VERBOSE = True



class accurate_classifier:

    def __init__(self,model_weight_path) -> None:

        age_model  = os.path.join(model_weight_path,"deploy_age.prototxt")
        age_weight = os.path.join(model_weight_path,"age_net.caffemodel")
        self.age_net = cv2.dnn.readNetFromCaffe(age_model,age_weight)

        gender_model = os.path.join(model_weight_path,"deploy_gender.prototxt")
        gender_weight = os.path.join(model_weight_path,"gender_net.caffemodel")
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_model,gender_weight)

        self.model_mean_val = (78.4263377603, 87.7689143744, 114.895847746)     

        ### Age and Gender label ###
        #
        #   age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)','(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']
        #   gender_list = ['Male', 'Female']
        #
        ############################

    def face_detect_recognize(self,img):
        face_loc = face_recognition.face_locations(img)
        face_embedding = face_recognition.face_encodings(img, face_loc)
        # print(face_recognition.face_distance( [ test[0] ], test[1]))

        return list(zip(face_loc,face_embedding))

    def age_gender_classifying(self,img):


        blob = cv2.dnn.blobFromImage(img, 1, (227, 227), self.model_mean_val, swapRB=False)
        
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = gender_preds.argmax()

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = age_preds.argmax()

        return gender,age


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



if __name__ == "__main__":
    # for test
    face_handler = accurate_classifier("./weight_file")

    test_img_1 = cv2.imread("../test_img/face1.jpg")
    test_img_2 = cv2.imread("../test_img/jungwoo_1.jpg")

    pos1, em1 = face_handler.face_detect_recognize(test_img_1)[0]
    pos2, em2 = face_handler.face_detect_recognize(test_img_2)[0]
    # #top, right, bottom, left
    print(face_recognition.face_distance([em1],em2))

    te1 = str_2_embedding (embedding_2_str(em1))
    te2 = str_2_embedding (embedding_2_str(em2))
    print(l2_norm(te1,te2))
    print(face_recognition.face_distance([np.array(te1)],np.array(te2)))

    # em_str = embedding_2_str(em)

    # print(em_str,len(em_str))
    # print(len(em),max(em),min(em))
    # print(str_2_embedding(em_str))
    
    # face_recognition.face_distance()

    # test_img = cv2.imread("face2.jpg")
    # print(face_handler.age_gender_classifying(test_img))
    # test_1 = cv2.imread("ris.jpg")
    # test_2 = cv2.imread("test0.jpg")
    # a = face_handler.face_detect_recognize(test_1)
    # b = face_handler.face_detect_recognize(test_2)

    # print(a[0][0],b[0][0])
    # #top, right, bottom, left
    # top, right, bottom, left = a[0][0]
    # face_1 = test_1[top:bottom,left:right].copy()
    # cv2.imwrite("face1.jpg",face_1)

    # top, right, bottom, left = b[0][0]
    # face_2 = test_2[top:bottom,left:right].copy()
    # cv2.imwrite("face2.jpg",face_2)
    # # print(a[0][1])
    # print(face_recognition.face_distance( [a[0][1] ], b[0][1] ))



