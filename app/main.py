import os
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

app = FastAPI()

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
CHECKPOINT_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet'

# Charger le modèle TensorFlow
configs = config_util.get_configs_from_pipeline_file(os.path.join(CHECKPOINT_PATH, 'pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

# Charger la carte de labels
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(CHECKPOINT_PATH, LABEL_MAP_NAME), use_display_name=True)

# Fonction de détection de plaque d'immatriculation
def detect_license_plate(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Détecter les contours dans l'image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours pour ceux qui pourraient être des plaques d'immatriculation
    plates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.5 <= aspect_ratio <= 5:
                plates.append((x, y, w, h))
    
    # Si aucune plaque n'est trouvée, retourner une chaîne vide
    if not plates:
        return ""
    
    # Région d'intérêt (ROI) pour la première plaque trouvée
    x, y, w, h = plates[0]
    roi = image[y:y+h, x:x+w]
    
    # Utiliser EasyOCR pour la détection de texte sur la plaque
    reader = easyocr.Reader(['en'])
    results = reader.readtext(roi)
    
    # Si aucun texte n'est trouvé, retourner une chaîne vide
    if not results:
        return ""
    
    # Extraire le texte de la première détection
    plate_text = results[0][1]
    
    return plate_text

# Route pour la détection
@app.post("/detection")
async def detect_license_plate_api(file: UploadFile = File(...)):
    # Lire l'image depuis la requête
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Exécuter la détection de plaque d'immatriculation
    plate_number = detect_license_plate(image)
    
    return JSONResponse(content={"plate_number": plate_number})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

