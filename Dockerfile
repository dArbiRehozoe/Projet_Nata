# Utilisez une image TensorFlow avec Jupyter préinstallé
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Copiez les fichiers nécessaires dans le conteneur
COPY . /app
WORKDIR /app

# Installez les dépendances supplémentaires, s'il y en a
RUN pip install -r requirements.txt

# Exposez le port de Jupyter Notebook
EXPOSE 8888

# Démarrez le serveur Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

