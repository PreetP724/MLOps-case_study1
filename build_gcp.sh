gcloud init
gcloud auth init
gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev
docker build -t app .
docker tag app:latest northamerica-northeast1-docker.pkg.dev/lofty-mark-419421/app/app:latest
docker push northamerica-northeast1-docker.pkg.dev/lofty-mark-419421/app/app:latest