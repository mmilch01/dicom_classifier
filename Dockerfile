FROM registry.nrg.wustl.edu/docker/nrg-repo/nrg-env:latest

RUN mkdir -p /usr/local/MATLAB_Runtime && \
	mkdir -p /usr/local/benice && \
	rm -rf /nrgpackages/tools/nrg-improc && \
	mkdir -p /docker_mount && \
	mkdir -p /input && \
	yum -y install java
	

COPY MATLAB_Runtime /usr/local/MATLAB_Runtime
COPY benice /usr/local/benice
COPY nrg-improc /nrgpackages/tools/nrg-improc

ENV FIV_HOME=/nrgpackages/tools/fiv \
	PATH=/nrgpackages/tools/fiv:/nrgpackages/tools/nrg-improc/Perceptron:$PATH

WORKDIR /docker_mount
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]

