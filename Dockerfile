FROM python:3.8-buster
RUN apt-get update && apt-get install -y build-essential

RUN mkdir /SDGym && \
    mkdir /SDGym/sdgym && \
    mkdir /SDGym/privbayes

# Copy code
COPY setup.py README.md HISTORY.md MANIFEST.in LICENSE Makefile setup.cfg /SDGym/
COPY /sdgym/ /SDGym/sdgym
COPY /privbayes/ /SDGym/privbayes

WORKDIR /SDGym

# Install project
RUN make install compile
ENV PRIVBAYES_BIN /SDGym/privbayes/privBayes.bin
ENV TF_CPP_MIN_LOG_LEVEL 2

CMD ["echo", "Usage: docker run -ti sdvproject/sdgym sdgym COMMAND OPTIONS"]
