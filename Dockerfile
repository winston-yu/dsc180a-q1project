ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

USER root
RUN apt update

RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
RUN conda install scikit-learn==1.0.2
RUN conda install pyg -c pyg

# wiyu/q1project:latest