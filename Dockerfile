# Utilisez une image TensorFlow avec Jupyter préinstallé
FROM tensorflow/tensorflow:latest-gpu-jupyter
# Spécifiez la version de Python# Mise à jour des packages et installation de Python 3.9
RUN apt-get update && apt-get install -y python3.9

# Installation des outils de dévelo
RUN apt-get install -y python3-dev
RUN apt-get update && apt-get install -y build-essential protobuf-compiler
RUN apt-get update && apt-get install -y  libatlas-base-dev  liblapack-dev gfortran libcairo2-dev libmariadb-dev build-essential libffi-dev curl
# Copiez les fichiers nécessaires dans le conteneur
COPY object_detection/protos /app/object_detection/protos
WORKDIR /app
COPY . /app
RUN python3 -m pip install --upgrade pip
RUN pip install opencv-python-headless \
                numpy \
                matplotlib
RUN pip install google-api-python-client immutabledict kaggle oauth2client opencv-python-headless py-cpuinfo sentencepiece seqeval tensorflow-datasets tensorflow-hub tensorflow-model-optimization tensorflow-text

RUN pip install --upgrade tensorflow

RUN pip install object_detection

RUN pip install tf-models-official

# Installez les dépendances supplémentaires, s'il y en a
RUN pip install -r requirements.txt
# Exposez le port de Jupyter Notebook
RUN protoc object_detection/protos/*.proto --python_out=.
EXPOSE 8888

# Démarrez le serveur Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
