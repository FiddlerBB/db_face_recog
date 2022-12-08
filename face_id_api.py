from modules import get_face_rec, get_encode, analyze_face, update_face, get_image_b64, get_face_rec, image_bgr
from scipy.spatial.distance import cosine
import cv2
import json


def update_user(image, directory_id):
    db_encoder, img = analyze_face(image)
    update_face(img, db_encoder, directoryId=directory_id)
    return img


def face_recognize(encoding_dict, image, recognition_init=0.6):
    base64_result = []
    id_result = []

    face, pt1, pt2 = get_face_rec(image)

    encode = get_encode(image=face)
    # id = 'unknown'
    distance = float('inf')

    for db_id, db_encode in encoding_dict.items():
        # print(db_id)
        dist = cosine(db_encode, encode)
        # print(dist)
        if dist < recognition_init:
            id = db_id
            distance = dist
            id_result.append(id)
            base64_result.append(get_image_b64(id))
    # img_origin = image_bgr(image)
    # if id == 'unknown':
    #     cv2.rectangle(img_origin, pt1, pt2, (0, 0, 255), 2)
    #     cv2.putText(img_origin, id, pt1,
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    # else:
    #     cv2.rectangle(img_origin, pt1, pt2, (0, 255, 0), 2)
    #     cv2.putText(img_origin, id + f'__{distance:.2f}', (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (0, 200, 200), 2)
    # cv2.imshow('a', img_origin)
    # cv2.waitKey(0)
            
    result_dict = dict(zip(id_result, base64_result))
    return result_dict
