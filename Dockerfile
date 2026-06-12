FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git zip unzip bzip2 ca-certificates\
    && rm -rf /var/lib/apt/lists/*

ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    PATH=/src:/usr/local/bin:/opt/micromamba/bin:/usr/bin:/bin \
    PYLIB=/src

RUN mkdir -p /opt/micromamba \
    && curl -L https://anaconda.org/conda-forge/micromamba/2.4.0/download/linux-64/micromamba-2.4.0-0.tar.bz2 \
       | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && chmod 0755 /usr/local/bin/micromamba && chmod -R a+rX /opt/micromamba \
    && mkdir -p /models/model_mirrir_1351062s_15Kt.10.04.2023 \
    && mkdir -p /models/model_fc_39374-600.03.20.2024 \
    && mkdir -p /output \
    && mkdir -p /input \
    && mkdir -p /resources

RUN micromamba install -y -n base -c conda-forge \
    python=3.8.16 \
    tensorflow==2.13 \
    pydicom==2.4.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    pandas==2.0.3 \
    Pillow==10.0.0 \
    requests \
    pip \
    pyxnat==1.6.3 \
    && micromamba clean -a -y

COPY --chmod=755 src /src
COPY --chmod=755 model_mirrir_1351062s_15Kt.10.04.2023 /models/model_mirrir_1351062s_15Kt.10.04.2023
COPY --chmod=755 model_fc_39374-600.03.20.2024 /models/model_fc_39374-600.03.20.2024
COPY --chmod=755 entrypoint.sh /entrypoint.sh

WORKDIR /output
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
