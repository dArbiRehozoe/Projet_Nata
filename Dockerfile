# Utilisez une image TensorFlow avec Jupyter préinstallé
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Ajouter la clé GPG du référentiel CUDA
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Mettre à jour les paquets en forçant l'utilisation d'HTTP pour éviter les erreurs GPG
RUN apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true

# Mettre à jour à nouveau la liste des paquets
RUN apt-get update --allow-releaseinfo-change

# Installer les autres dépendances nécessaires
RUN apt-get update && apt-get install -y \
    pkg-config \
    libfuse-dev \
    python3.9

# Continuer avec l'installation des bibliothèques CUDA
RUN apt-get install -y --no-install-recommends \
    cuda \
    libcudnn8 \
    libcudnn8-dev \
    libnccl2 \
    libnccl-dev

# Installation des outils de développement
RUN apt-get install -y python3-dev
RUN apt-get update && apt-get install -y build-essential protobuf-compiler
RUN apt-get update && apt-get install -y  libatlas-base-dev  liblapack-dev gfortran libcairo2-dev libmariadb-dev build-essential libffi-dev curl

# Copiez les fichiers nécessaires dans le conteneur

WORKDIR /app
COPY . /app

# Installation des bibliothèques Python
RUN python3 -m pip install --upgrade pip
RUN pip install opencv-python-headless \
                numpy \
                matplotlib \
                google-api-python-client immutabledict kaggle oauth2client opencv-python-headless py-cpuinfo sentencepiece seqeval tensorflow-datasets tensorflow-hub tensorflow-model-optimization tensorflow-text

# Installer Jupyter et ses dépendances
RUN pip install -r requirements.txt
RUN pip install jupyter
RUN pip install --upgrade protobuf
# Installer les bibliothèques liées à l'object detection
RUN pip install object_detection
RUN pip install tf-models-official

# Exécutez la compilation des fichiers protobuf
# Get protoc 3.0.0, rather than the old version already in the container
RUN rm -rf models
RUN git clone https://github.com/tensorflow/models.git

# Run protoc on the object detection repo
RUN cd models/research && \
	protoc --python_out=. ./object_detection/protos/*.proto

EXPOSE 8888

# Démarrez le serveur Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

