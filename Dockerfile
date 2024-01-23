# Utilisez une image TensorFlow avec Jupyter préinstallé
FROM tensorflow/tensorflow:latest-gpu-jupyter
# Spécifiez la version de Python# Mise à jour des packages et installation de Python 3.9
RUN apt-get update && apt-get install -y python3.9

# Installation des outils de dévelo
RUN apt-get install -y python3-dev
RUN apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get install -y  libatlas-base-dev  liblapack-dev gfortran libcairo2-dev libmariadb-dev build-essential libffi-dev curl
# Copiez les fichiers nécessaires dans le conteneur
WORKDIR /app
COPY . /app
RUN python3 -m pip install --upgrade pip
RUN pip install opencv-python-headless \
                numpy \
                matplotlib

# Installez les dépendances supplémentaires, s'il y en a
RUN pip install -r requirements.txt
# Exposez le port de Jupyter Notebook
EXPOSE 8888

# Démarrez le serveur Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

