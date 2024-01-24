# Utilisez une image TensorFlow avec Jupyter préinstallé
FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

# Ajouter la clé GPG du référentiel CUDA
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Mettre à jour les paquets en forçant l'utilisation d'HTTP pour éviter les erreurs GPG
RUN apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true

# Mettre à jour à nouveau la liste des paquets
RUN apt-get update --allow-releaseinfo-change

RUN apt-get update && apt-get install -y \
    pkg-config \
    libfuse-dev

RUN apt-get update && apt-get install -y python3.9
# Continuer avec l'installation des bibliothèques CUDA
RUN apt-get install -y --no-install-recommends \
    cuda-compiler-11-4 \
    libcudnn8=8.2.4.15-1+cuda11.4 \
    libcudnn8-dev=8.2.4.15-1+cuda11.4 \
    libnccl2=2.11.4-1+cuda11.4 \
    libnccl-dev=2.11.4-1+cuda11.4 \
    && apt-mark hold libcudnn8 \
    && apt-mark hold libnccl2
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


RUN pip install object_detection

RUN pip install tf-models-official

# Installez les dépendances supplémentaires, s'il y en a
RUN pip install --no-cache-dir -r requirements.txt
# Exécutez la compilation des fichiers protobuf
RUN protoc object_detection/protos/*.proto --python_out=.
EXPOSE 8888

# Démarrez le serveur Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
