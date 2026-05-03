az login
az acr login --name appregistrymlops 
DOCKER_BUILDKIT=0 docker build -t app .
docker tag app:latest appregistrymlops.azurecr.io/app:latest
docker push appregistrymlops.azurecr.io/app:latest