docker run \
    --gpus all\
    -it \
    --rm \
    -p 8888:888 \
    --shm-size 8G \
    --volume="$(pwd):$(pwd)" \
    --workdir="$(pwd)" \
    ofrin/tensorflow_2.2_custom:latest

xhost local:root
docker run \
	-it \
	--rm \
	--cpuset-cpus 0-6 \
	--gpus '"device=0"' \
	--shm-size 8G \
	--volume "$(pwd):$(pwd)" \
	--volume "/home/sanghyunjo/Desktop/Tensorflow_2_Tutorials/:/home/sanghyunjo/Desktop/Tensorflow_2_Tutorials/" \
	--volume "/tmp/.X11-unix:/tmp/.X11-unix:ro" \
	--volume "/dev/snd:/dev/snd" \
	-e DISPLAY=unix$DISPLAY \
    --device=/dev/video0 \
	--workdir "$(pwd)" \
	ofrin/tensorflow_2.2_custom:latest \
    python generate_tfrecords.py

