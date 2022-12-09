from modules import load_predict_db, process_input
from fastapi import File, FastAPI
from face_id_api import update_user, face_recognize

app = FastAPI(title='Testing FastAPI')


@app.post("/face-id/updated-data/", tags=['Update User Face data'])
async def get_body(User_image: bytes = File(...), Directory_id: int = File(...)):
    input_image = process_input(User_image)
    img = update_user(input_image, directory_id=Directory_id)
    if img is not None:
        return {
            'check_update': '200 OK'
        }
    return {
        'check_update': '404 ERROR'
    }


@app.post("/face-id/face-recognition/", tags=['Recognize User Face by User ID'])
async def get_body(New_image: bytes = File(...), Directory_id: int = File(...)):
    input_image = process_input(New_image)
    result = face_recognize(image=input_image, encoding_dict=load_predict_db(
        directoryId=Directory_id), recognition_init=0.6)
    return {
        'count_users': len(result),
        'directory_id': Directory_id,
        'user_id': result
    }
