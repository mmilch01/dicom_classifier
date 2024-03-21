FROM centos:7.5.1804

#RUN rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
#RUN yum -y install https://www.elrepo.org/elrepo-release-7.0-4.el7.elrepo.noarch.rpm
RUN yum -y install epel-release


RUN yum -y install bc wget curl epel-release which tree
RUN yum -y install CharLS libtiff libXfont zip
#RUN yum -y install zip unzip ImageMagick html2ps xvfb bc wget epel-release bzip2 which git cmake gcc gcc-c++ libstdc++-static epel-release csh compat-libgfortran-41 libjpeg-turbo-utils


ENV PATH=/src:/nrgpackages/packages/tensorflow2.13-cpu/bin:$PATH

RUN cd /tmp && wget https://repo.continuum.io/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh; chmod +x Miniconda3-py38_23.5.2-0-Linux-x86_64.sh && \
     ./Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -u -b -p /nrgpackages/packages/tensorflow2.13-cpu

RUN cd /tmp && \
	wget http://li.nux.ro/download/nux/dextop/el7/x86_64/dcmtk-3.6.0-16.el7.nux.x86_64.rpm && \
	rpm -Uvh dcmtk-3.6.0-16.el7.nux.x86_64.rpm

RUN pip install --upgrade pip
#workariunds to avoid runtime errors.
RUN pip install certifi --ignore-installed

#install python packages

RUN pip install tensorflow-cpu==2.13
RUN pip install pydicom scikit-learn ipywidgets matplotlib

#RUN yum -y install zip unzip ImageMagick html2ps xvfb bc wget epel-release bzip2 which git cmake gcc gcc-c++ libstdc++-static epel-release csh compat-libgfortran-41 libjpeg-turbo-utils
#RUN yum -y install CharLS libXfont gnuplot xorg-x11-server-Xvfb 

RUN mkdir -p /nrgpackages/packages/tensorflow2.13-cpu
RUN mkdir -p /models/model_mirrir_1351062s_15Kt.10.04.2023
RUN mkdir -p /models/model_fc_39374-600.03.20.2024
RUN mkdir -p /output
RUN mkdir -p /input

COPY src /src
COPY model_mirrir_1351062s_15Kt.10.04.2023 /models/model_mirrir_1351062s_15Kt.10.04.2023
COPY model_fc_39374-600.03.20.2024 /models/model_fc_39374-600.03.20.2024

ENV PYLIB=/src

RUN rm -rf /tmp/* /var/cache/yum

WORKDIR /output

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
