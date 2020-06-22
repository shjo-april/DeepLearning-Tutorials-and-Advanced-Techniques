docker build -t ofrin/tensorflow:v2.2 .
docker tag d58 ofrin/tensorflow_2.2_custom:latest

docker login

docker push <hub-user>/<repo-name>:<tag>
docker push ofrin/tensorflow_2.2_custom

docker pull ofrin/tensorflow_2.2_custom