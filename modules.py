import cv2
import mtcnn
import base64
import sqlite3
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import Normalizer
from architecture import InceptionResNetV2
from typing import Tuple

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


def normalize(img: bytes) -> bytes:
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def get_detector():
    face_detector = mtcnn.MTCNN()
    return face_detector


def image_bgr(image) -> np.array:
    img_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return img_bgr


def image_bgr2rgb(image) -> np.array:
    img_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# convert cv2 to base64
def to_base64(img: np.array) -> base64:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf)


# convert base64 cv2
def from_base64(buf: base64) -> np.array:
    buf_decode = base64.b64decode(buf)
    buf_arr = np.frombuffer(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


# convert byte to np.array
def byte2array(byte, dtype=np.float32) -> np.array:
    array_return = np.frombuffer(byte, dtype=np.float32)
    return array_return


# process image
def get_face(image) -> Tuple[np.array, int, int]:
    img_RGB = image_bgr2rgb(image)
    # b64_img = to_base64(img_origin)
    face_detector = get_detector()
    x = face_detector.detect_faces(img_RGB)
    x1, y1, width, height = x[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
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
    face, pt1, pt2 = get_face(img)
    img_b64 = to_base64(face)
    encoder = get_encode(model_path=MODEL_PATH, image=face)
    encodes.append(encoder)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode_input = l2_normalizer.transform(
            np.expand_dims(encode, axis=0))[0]
    return encode_input, img_b64


# query data
def select_by_id(id: int) -> dict:
    df = pd.read_sql(f"select * from faceid_table where id = {id}", conn)
    stored_predict = df['predict_result'][0]
    stored_predict = byte2array(stored_predict)
    return stored_predict


# insert new user
def insert_user(b64_img, encoded_input: bytes, directoryId: int) -> None:
    c.execute("INSERT INTO faceid_table (image_base64, predict_result, directoryId) VALUES (?,?,?)",
              (b64_img, encoded_input, directoryId))
    conn.commit()


# scan db and get users by directoryId
def load_predict_db(directoryId: int) -> dict:
    df = pd.read_sql(
        f"select id, predict_result from faceid_table where directoryId={directoryId}", conn)
    stored_predict = df['predict_result'].apply(byte2array)
    id = df['id'].apply(str)
    dict_result = dict(zip(id, stored_predict))
    return dict_result

# select base64 format based on specific id
def get_image_b64(id: int) -> base64:
    df = pd.read_sql(
        f"select image_base64 from faceid_table where id = {id}", conn)
    image_b64 = df['image_base64'][0]
    return image_b64


# process file input
def process_input(file) -> cv2:
    image = io.BytesIO(file)
    input_image = np.asarray(bytearray(image.read()), dtype=np.uint8)
    return input_image

if __name__ == '__main__':
    pass
    df = pd.read_sql(f"select * from faceid_table", conn)
    print(df)

    # image = b642rgb(id=1)
    # cv2.imshow('s', image)
    # cv2.waitKey(0)
    # print(image)
