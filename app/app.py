import os
import sys
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
sys.path.append('./Tensorflow/models/official')
app = FastAPI()

# Configuration du modèle et chargement des poids du modèle
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
CHECKPOINT_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet'

# Charger le modèle
configs = config_util.get_configs_from_pipeline_file(os.path.join(CHECKPOINT_PATH, 'pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

# Charger la carte de labels
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(CHECKPOINT_PATH, LABEL_MAP_NAME), use_display_name=True)

# Fonction de détection de plaque d'immatriculation
def detect_license_plate(image):
    # Votre logique de détection de plaque d'immatriculation
    return "XXX 1234"  # Remplacez ceci par les vrais résultats de détection

# Route pour la détection
@app.post("/detection")
async def detect_license_plate_api(file: UploadFile = File(...)):
    # Lire l'image depuis la requête
    contents = await file.read()
    image = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Exécuter la détection
    plate_number = detect_license_plate(image)
    
    if plate_number:
        return JSONResponse(content={"plate_number": plate_number})
    else:
        return JSONResponse(content={"plate_number": False})

@app.get("/")
async def read_root():
    return {"message": "Hello World"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

