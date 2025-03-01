#!/bin/bash

# Variables
ECR_URL="205930621695.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_NAME="rag_lambda"
TAG="latest"

# Construir la imagen de Docker
echo "Construyendo la imagen de Docker..."
docker build -t $IMAGE_NAME .

# Etiquetar la imagen para ECR
echo "Etiquetando la imagen..."
docker tag $IMAGE_NAME:latest $ECR_URL/$IMAGE_NAME:$TAG

# Autenticarse en AWS ECR
echo "Autenticando en AWS ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URL

# Subir la imagen a ECR
echo "Pushing la imagen a AWS ECR..."
docker push $ECR_URL/$IMAGE_NAME:$TAG

echo "Despliegue completado!"
