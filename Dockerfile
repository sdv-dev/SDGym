FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
CMD nvidia-smi

RUN apt-get update && apt-get install -y build-essential && apt-get -y install curl
RUN apt-get -y install python3.8 python3-distutils && ln -s /usr/bin/python3.8 /usr/bin/python
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
RUN make install-all compile
RUN pip install -U numpy==1.20
ENV PRIVBAYES_BIN /SDGym/privbayes/privBayes.bin
ENV TF_CPP_MIN_LOG_LEVEL 2

CMD ["echo", "Usage: docker run -ti sdvproject/sdgym sdgym COMMAND OPTIONS"]
