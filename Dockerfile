FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

RUN apt-get update && apt-get install -y build-essential curl python3.7 python3.7-dev \
    python3-distutils && ln -s /usr/bin/python3.7 /usr/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && ln -s /usr/bin/pip3 /usr/bin/pip

RUN mkdir /SDGym && \
    mkdir /SDGym/sdgym && \
    mkdir /SDGym/privbayes

# Copy code
COPY setup.py README.md HISTORY.md MANIFEST.in LICENSE Makefile setup.cfg /SDGym/
COPY /sdgym/ /SDGym/sdgym
COPY /privbayes/ /SDGym/privbayes

WORKDIR /SDGym

# Install project
RUN pip install ydata-synthetic==0.6.1
RUN pip install .[gretel] --no-binary pomegranate
RUN make compile
ENV PRIVBAYES_BIN /SDGym/privbayes/privBayes.bin
ENV TF_CPP_MIN_LOG_LEVEL 2

CMD ["echo", "Usage: docker run -ti sdvproject/sdgym sdgym COMMAND OPTIONS"]
