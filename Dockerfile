# Utilisation de l'image officielle Python 3.9 comme image de base
FROM python:3.9-slim

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn httpx opencv-python-headless tensorflow==2.7.0 protobuf==3.20.0 easyocr object_detection  easyocr numpy tf_slim tensorflow_io python-multipart tf-models-official
# Copie de l'application dans le conteneur
COPY . /app

# Définition du répertoire de travail
WORKDIR /app/app


# Exposition du port 8080
EXPOSE 8080

# Commande pour lancer l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

