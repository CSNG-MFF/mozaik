FROM python:3.7-buster as packages
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
        libjpeg-dev \
        liblapack-dev \
        libncurses5-dev \
        libopenmpi-dev \
        libpng++-dev \
        libreadline-dev \
        openmpi-bin \
        pkg-config \
        subversion \
        wget \
        zlib1g-dev \
 && pip install pipenv==2020.8.13

WORKDIR /source
COPY Pipfile Pipfile.lock ./
ARG PACKAGES_DIR=/source/packages
# six is not installed into the prefix directory for some reason
RUN PIP_PREFIX=${PACKAGES_DIR} \
    PIP_IGNORE_INSTALLED=1 \
    pipenv install --system --ignore-pipfile --deploy \
 && pip install --prefix=${PACKAGES_DIR} --ignore-installed six
ENV PATH=${PATH}:${PACKAGES_DIR}/bin
ENV PYTHONPATH=${PACKAGES_DIR}/lib/python3.7/site-packages:${PYTHONPATH}

RUN wget https://github.com/nest/nest-simulator/archive/v2.20.0.tar.gz \
 && tar xvfz v2.20.0.tar.gz \
 && cd nest-simulator-2.20.0 \
 && cmake \
        -DCMAKE_INSTALL_PREFIX:PATH=/opt/nest \
        -Dwith-boost=ON \
        -Dwith-mpi=OFF \
        -Dwith-optimize='-O3' \
        ./ \
 && make \
 && make install

WORKDIR /source/mozaik
COPY . ./
RUN pip install --prefix=${PACKAGES_DIR} .


FROM python:3.7-slim-buster as prod
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libfreetype6=2.9.1-3+deb10u1 \
        libgomp1=8.3.0-6 \
        libgsl23=2.5+dfsg-6 \
        libjpeg62-turbo=1:1.5.2-2+b1 \
        libncurses5=6.1+20181013-2+deb10u2 \
        libtk8.6=8.6.9-2 \
        openmpi-bin=3.1.3-11 \
        ssh=1:7.9p1-10+deb10u2 \
 && rm -rf /var/lib/apt/lists/*

ARG PACKAGES_DIR=/source/packages
COPY --from=packages ${PACKAGES_DIR} /usr/local
COPY --from=packages /opt/nest /opt/nest
ENV PYTHONPATH=/opt/nest/lib/python3.7/site-packages:$PYTHONPATH

RUN groupadd -g 1000 mozaik \
 && useradd -m -u 1000 -g mozaik mozaik

USER mozaik
WORKDIR /app
ENTRYPOINT ["python"]


FROM prod as dev
USER root
RUN apt-get update \
 && apt-get install -y \
        git \
 && pip install pipenv==2020.8.13

WORKDIR /app
RUN chown -R mozaik:mozaik .
COPY --chown=mozaik:mozaik Pipfile Pipfile.lock ./
RUN pipenv install --system --ignore-pipfile --deploy --dev

COPY --chown=mozaik:mozaik . ./
RUN pip install -e .

USER mozaik
ENV SHELL /bin/bash
ENTRYPOINT [""]
