import cv2
import numpy as np
import base64
import sqlite3 
import pandas as pd
import numpy as np


conn = sqlite3.connect('db/faceID.db')
c = conn.cursor()

# convert cv2 to base64
def to_base64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf)

# convert base64 cv2
def from_base64(buf):
    buf_decode = base64.b64decode(buf)
    buf_arr = np.frombuffer(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


def byte2array(byte):
    array_return = np.frombuffer(byte, dtype=np.float32)
    return array_return

# create table
# c.execute('''
# CREATE table if not exists faceid_table (
#     id INTEGER PRIMARY KEY AUTOINCREMENT, image_base64 text, predict_result text
# )
# '''
# )

def select_by_id(id):
    df = pd.read_sql(f"select * from faceid_table where id = {id}", conn)
    stored_predict = df['predict_result'][0]
    stored_predict = byte2array(stored_predict)
    return stored_predict

def update_face(b64_img, encoded_input):
    c.execute("INSERT INTO faceid_table (image_base64, predict_result) VALUES (?,?)", (b64_img, encoded_input))
    conn.commit()

def load_predict_db() -> dict: 
    df = pd.read_sql("select * from faceid_table", conn)
    all_stored_predict = df['predict_result']
    all_stored_predict = [byte2array(i) for i in all_stored_predict]
    id = df['id'].apply(str).tolist()
    dict_result = dict(zip(id, all_stored_predict))
    return dict_result

# df = pd.read_sql(f"select * from faceid_table", conn)
# print(df)