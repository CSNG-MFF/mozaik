FROM ubuntu:18.04 as packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y \
        cmake \
        g++ \
        gfortran \
        git \
        libblas-dev \
        libfreetype6-dev \
        libgsl0-dev \
        libjpeg8-dev \
        liblapack-dev \
        libncurses5-dev \
        libopenmpi-dev \
        libpng++-dev \
        libreadline-dev \
        openmpi-bin \
        python-dev \
        python-nose \
        python-pip \
        python-tk \
        subversion \
        zlib1g-dev \
        wget

RUN pip install --upgrade distribute \
 && pip install \
        cython \
        interval \
        lazyarray \
        matplotlib==2.1.1 \
        mpi4py \
        neo==0.5.2 \
        numpy \
        param==1.5.1 \
        parameters \
        Pillow \
        psutil \
        pynn \
        quantities \
        scipy

WORKDIR /source
RUN git clone https://github.com/antolikjan/imagen.git \
 && cd imagen \
 && python setup.py install

RUN wget https://github.com/nest/nest-simulator/archive/v2.20.0.tar.gz \
 && tar xvfz v2.20.0.tar.gz \
 && cd nest-simulator-2.20.0 \
 && cmake \
        -Dwith-mpi=OFF \
        -Dwith-boost=ON \
        -DCMAKE_INSTALL_PREFIX:PATH=/opt/nest \
        -Dwith-optimize='-O3' \
        ./ \
 && make \
 && make install

WORKDIR /source/mozaik
COPY mozaik ./mozaik
COPY setup.py README.rst ./
RUN python setup.py install


FROM ubuntu:18.04 as prod
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libfreetype6 \
        libgomp1 \
        libgsl23 \
        libjpeg8 \
        libncurses5 \
        libpython2.7 \
        openmpi-bin \
        python-nose \
        python-six \
        python-tk \
        python2.7 \
        ssh \
 && rm -rf /var/lib/apt/lists/*

ARG PACKAGES_DIR=/usr/local/lib/python2.7/dist-packages
COPY --from=packages $PACKAGES_DIR $PACKAGES_DIR
COPY --from=packages /opt/nest /opt/nest
ENV PYTHONPATH=/opt/nest/lib/python2.7/site-packages

RUN groupadd -g 1000 mozaik \
 && useradd -m -u 1000 -g mozaik mozaik

USER mozaik
WORKDIR /app
ENTRYPOINT ["python"]

FROM prod as dev
USER root
RUN apt-get update \
 && apt-get install -y \
        python-pip \
 && pip install pytest

USER mozaik
ENTRYPOINT [""]
