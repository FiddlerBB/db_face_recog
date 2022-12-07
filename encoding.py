from architecture import * 
import cv2
import mtcnn
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
from modules import to_base64, from_base64, load_predict_db, update_face, byte2array

######pathsandvairables#########
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


image_path = 'test_data/disaster-girl.jpg'
def process_encode_face(path):
    img_BGR = cv2.imread(path)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    b64_img = to_base64(img_BGR)

    x = face_detector.detect_faces(img_RGB)
    x1, y1, width, height = x[0]['box']
    x1, y1 = abs(x1) , abs(y1)
    x2, y2 = x1+width , y1+height
    face = img_RGB[y1:y2 , x1:x2]

    face = normalize(face)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    encode = face_encoder.predict(face_d)[0]
    encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode_input = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
    return b64_img, encode_input

#insert image to db
b64_img, encode_input = process_encode_face(image_path)
update_face(b64_img, encode_input)

# load all predict result
# all_predict_result = load_predict_db()


if __name__ == '__main__':
    path = image_path