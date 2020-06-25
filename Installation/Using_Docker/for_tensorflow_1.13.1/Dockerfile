FROM tensorflow/tensorflow:1.13.1-gpu-py3

# install OpenCV
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get upgrade -y
RUN pip3 install opencv-python
RUN apt-get install python-opencv -y

# install PIL(pillow)
RUN pip3 install pillow

# install flask
RUN pip3 install flask

# install waitress
RUN pip3 install waitress

# install matplotlib
RUN pip3 install matplotlib

# install sklearn
RUN pip3 install sklearn

# install pydencecrf
RUN pip3 install pydensecrf

# install mtcnn
RUN pip3 install mtcnn