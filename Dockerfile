FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
CMD nvidia-smi

RUN apt-get update && apt-get install -y build-essential curl python3.9 python3-pip \
    python3-distutils && ln -s /usr/bin/python3.9 /usr/bin/python

RUN mkdir /SDGym && \
    mkdir /SDGym/sdgym && \

# Copy code
COPY pyproject.toml README.md HISTORY.md MANIFEST.in LICENSE Makefile setup.cfg /SDGym/
COPY /sdgym/ /SDGym/sdgym

WORKDIR /SDGym

# Install project
RUN pip install . --no-binary pomegranate
RUN make compile
ENV TF_CPP_MIN_LOG_LEVEL 2

CMD ["echo", "Usage: docker run -ti sdvproject/sdgym sdgym COMMAND OPTIONS"]
