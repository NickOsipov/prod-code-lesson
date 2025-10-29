# local build and push
docker build -t nickosipov/ml-service:latest .
docker push nickosipov/ml-service:latest

# vm
docker pull nickosipov/ml-service:latest
docker stop ml-service
docker rm ml-service
docker run -d -p 5000:5000 --name ml-service nickosipov/ml-service:latest