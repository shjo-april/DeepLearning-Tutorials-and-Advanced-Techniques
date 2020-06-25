FROM tensorflow/tensorflow:latest-gpu

# install OpenCV
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get upgrade -y

RUN pip3 install opencv-python
RUN apt-get install python-opencv -y

# install PIL(pillow)
RUN pip3 install pillow

# install matplotlib
RUN pip3 install matplotlib

# install sklearn
RUN pip3 install sklearn

# install pydencecrf
RUN pip3 install pydensecrf
