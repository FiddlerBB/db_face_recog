import cv2
import mtcnn
import base64
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from architecture import InceptionResNetV2

SHAPE = (160, 160)
MODEL_PATH = 'facenet_keras_weights.h5'
l2_normalizer = Normalizer('l2')

# connect DB
conn = sqlite3.connect('db/faceID.db')
c = conn.cursor()

# create table
c.execute('''
CREATE table if not exists faceid_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT, image_base64 text, predict_result text, directoryId int
)
'''
          )


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def get_detector():
    face_detector = mtcnn.MTCNN()
    return face_detector


def image_bgr(image):
    img_BGR = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img_BGR


def image_bgr2rgb(img_origin):
    img_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    return img_rgb


def b64_rgb(id: int) -> np.array:
    image = get_image_b64(id)
    image_array = from_base64(image)
    return image_array


# convert cv2 to base64
def to_base64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf)


# convert base64 cv2
def from_base64(buf):
    buf_decode = base64.b64decode(buf)
    buf_arr = np.frombuffer(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


# convert byte to np.array
def byte2array(byte, dtype=np.float32):
    array_return = np.frombuffer(byte, dtype=np.float32)
    return array_return


# process image
def get_location(image):
    # img_BGR = cv2.imread(image)
    img_origin = image_bgr(image)
    img_RGB = image_bgr2rgb(img_origin)
    # b64_img = to_base64(img_origin)
    face_detector = get_detector()
    x = face_detector.detect_faces(img_RGB)
    x1, y1, width, height = x[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    return x1, y1, x2, y2, img_RGB


def get_face_rec(path):
    x1, y1, x2, y2, img_RGB = get_location(path)
    face = img_RGB[y1:y2, x1:x2]
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    return face, pt1, pt2


def get_encode(image, model_path=MODEL_PATH, required_shape=SHAPE):
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(model_path)
    normalize_face = normalize(image)
    face = cv2.resize(normalize_face, required_shape)
    face_id = np.expand_dims(face, axis=0)
    encode = face_encoder.predict(face_id)[0]
    return encode


def analyze_face(img):
    encodes = []
    # img_origin = image_bgr(img)
    face, pt1, pt2 = get_face_rec(img)
    img_b64 = to_base64(face)
    encoder = get_encode(model_path=MODEL_PATH, image=face)
    encodes.append(encoder)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode_input = l2_normalizer.transform(
            np.expand_dims(encode, axis=0))[0]
    return encode_input, img_b64


# convert cv2 to base64
def to_base64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf)


# convert base64 cv2
def from_base64(buf):
    buf_decode = base64.b64decode(buf)
    buf_arr = np.frombuffer(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


# convert byte to np.array
def byte2array(byte, dtype=np.float32):
    array_return = np.frombuffer(byte, dtype=np.float32)
    return array_return


# query data
def select_by_id(id):
    df = pd.read_sql(f"select * from faceid_table where id = {id}", conn)
    stored_predict = df['predict_result'][0]
    stored_predict = byte2array(stored_predict)
    return stored_predict


def update_face(b64_img, encoded_input: bytes, directoryId: int) -> None:
    c.execute("INSERT INTO faceid_table (image_base64, predict_result, directoryId) VALUES (?,?,?)",
              (b64_img, encoded_input, directoryId))
    conn.commit()


def load_predict_db(directoryId) -> dict:
    df = pd.read_sql(
        f"select id, predict_result from faceid_table where directoryId={directoryId}", conn)
    all_stored_predict = df['predict_result']
    all_stored_predict = [byte2array(i) for i in all_stored_predict]
    id = df['id'].apply(str).tolist()
    dict_result = dict(zip(id, all_stored_predict))
    return dict_result


def get_image_b64(id: int) -> base64:
    if id != 'unknown':
        df = pd.read_sql(
            f"select image_base64 from faceid_table where id = {id}", conn)
        image_b64 = df['image_base64'][0]
        return image_b64
    return "404 ERROR"


if __name__ == '__main__':
    pass
    df = pd.read_sql(f"select * from faceid_table", conn)
    print(df)

    # image = b642rgb(id=1)
    # cv2.imshow('s', image)
    # cv2.waitKey(0)
    # print(image)
