docker build -t ofrin/tensorflow_1.13.1_custom:latest .
docker tag d58 ofrin/tensorflow_1.13.1_custom:latest

docker login

docker push <hub-user>/<repo-name>:<tag>
docker push ofrin/tensorflow_1.13.1_custom

docker pull ofrin/tensorflow_1.13.1_custom

