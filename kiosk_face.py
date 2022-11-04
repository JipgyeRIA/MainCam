import face_recognition
import math
import cv2
from deepface import DeepFace


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


def face_checker(cam_stream, face_size_theshold = 0.0):
    ret, frame = cam_stream.read()

    if ret:
        H,W,_ = frame.shape
        cam_size = H*W
        face_loc = face_recognition.face_locations(frame)
        # if len(face_loc) != 0:
        #     top, right, bottom, left = face_loc[0]
        #     cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))
        # return 1,frame,""

        if len(face_loc) == 0:
            return -1, None, ""
        else:
            max_face_size = -1
            max_face_loc = []

            for top, right, bottom, left in face_loc:
                face_size = (bottom-top) * (right- left)
                if max_face_size < face_size:
                    max_face_size = face_size
                    max_face_loc = [top, right, bottom, left]

            if max_face_size < cam_size * face_size_theshold:
                return -1, None, ""

            else:
                top, right, bottom, left = max_face_loc
                print(top,right,bottom,left)
                cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))
                face_embedding = face_recognition.face_encodings(frame, [max_face_loc])
                return 1, frame, embedding_2_str(face_embedding[0])

    else:
        return -1, None, ""


def face_age_gender_checker(cam_stream, margin = 30):

    def age_labeler(input):
        if input < 6:
            return 0
        elif input < 38:
            return 1
        elif input < 60:
            return 2
        else:
            return 3

    ret, frame = cam_stream.read()
    if ret:
        H,W,_ = frame.shape
        cam_size = H*W
        face_loc = face_recognition.face_locations(frame)
        # if len(face_loc) != 0:
        #     top, right, bottom, left = face_loc[0]
        #     cv2.rectangle(frame, [left,top],[right,bottom],(0,255,0))
        # return 1,frame,""

        if len(face_loc) == 0:
            return -1, None,None
        else:
            answer = []
            for top, right, bottom, left in face_loc:
                im_t,im_b,im_l,im_r = max(0,top-margin),min(H,bottom+margin),max(0,left-margin),min(W,right+margin)

                response=DeepFace.analyze(frame[im_t:im_b,im_l:im_r],
                    actions=["gender","age"],
                    enforce_detection=False,
                    prog_bar= False,
                    detector_backend="dlib")
                answer.append([age_labeler(response["age"]), response["gender"]])

            return len(answer),frame, answer


    else:
        return -1, None,None



def get_single_face(cam):
    if cam.isOpened() == False:
        return -1
    
    time = 0
    while time < 100:
        ret,frame, emb = face_checker(cam)
        if ret != -1:
            break
    else:
        return -1

    return emb if emb != None else -1




if __name__ =='__main__':
    cam = cv2.VideoCapture(0)
    if cam.isOpened() == False:
            exit()
    

    while cv2.waitKey(33) < 0:
        ret, frame,answer = face_age_gender_checker(cam)
        print(answer)
        if ret != -1:
            cv2.imshow("VideoFrame", frame)
            cv2.waitKey(33)
            break
        #break

    #val,frame,emb =face_checker(cam)


