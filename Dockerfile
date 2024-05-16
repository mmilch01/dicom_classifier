FROM centos:7.5.1804

#RUN rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
#RUN yum -y install https://www.elrepo.org/elrepo-release-7.0-4.el7.elrepo.noarch.rpm
RUN yum -y install epel-release


RUN yum -y install bc wget curl epel-release which tree nodejs npm git
RUN yum -y install CharLS libtiff libXfont zip
#RUN yum -y install zip unzip ImageMagick html2ps xvfb bc wget epel-release bzip2 which git cmake gcc gcc-c++ libstdc++-static epel-release csh compat-libgfortran-41 libjpeg-turbo-utils


ENV PATH=/src:/nrgpackages/packages/tensorflow2.13-cpu/bin:$PATH

RUN cd /tmp && wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh; chmod +x Miniconda3-py38_23.5.2-0-Linux-x86_64.sh && \
     ./Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -u -b -p /nrgpackages/packages/tensorflow2.13-cpu

RUN cd /tmp && \
	wget http://li.nux.ro/download/nux/dextop/el7/x86_64/dcmtk-3.6.0-16.el7.nux.x86_64.rpm && \
	rpm -Uvh dcmtk-3.6.0-16.el7.nux.x86_64.rpm

RUN pip install --upgrade pip
#workariunds to avoid runtime errors.
RUN pip install certifi --ignore-installed

#install python packages

RUN pip install tensorflow-cpu==2.13 pydicom==2.4.3 scikit-learn==1.3.0 \
    ipywidgets==8.1.0 matplotlib==3.7.2 jupyterhub==3.0.0 pandas==2.0.3 \
    Pillow==10.0.0 ipydatagrid==1.2.1 voila==0.4.4 jhsingle-native-proxy==0.8.2 jupyterlab==3.6.7
    
RUN jupyter lab build --minimize=False

# Will be used to start as non-root user
COPY --from=tianon/gosu /gosu /usr/local/bin/
COPY entrypoint.sh /usr/local/bin/

# Copy in startup scripts form Jupyter Stacks. Should provide flexibility to start Juptyer notebook, lab, or in Hub mode.
COPY start-singleuser.py start-singleuser.sh start-notebook.sh start-notebook.py /usr/local/bin/

#RUN yum -y install zip unzip ImageMagick html2ps xvfb bc wget epel-release bzip2 which git cmake gcc gcc-c++ libstdc++-static epel-release csh compat-libgfortran-41 libjpeg-turbo-utils
#RUN yum -y install CharLS libXfont gnuplot xorg-x11-server-Xvfb 

RUN mkdir -p /nrgpackages/packages/tensorflow2.13-cpu
RUN mkdir -p /models/model_mirrir_1351062s_15Kt.10.04.2023
RUN mkdir -p /models/model_fc_39374-600.03.20.2024
RUN mkdir -p /output
RUN mkdir -p /input
RUN mkdir -p /resources

COPY src /src
COPY model_mirrir_1351062s_15Kt.10.04.2023 /models/model_mirrir_1351062s_15Kt.10.04.2023
COPY model_fc_39374-600.03.20.2024 /models/model_fc_39374-600.03.20.2024

ENV PYLIB=/src

RUN rm -rf /tmp/* /var/cache/yum

WORKDIR /output

ENTRYPOINT [ "entrypoint.sh"]
CMD ["start-notebook.sh"]
