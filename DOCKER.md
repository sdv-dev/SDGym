We support using Docker to run SDGym. This can help you avoid dependency issues when running SDGym
on Windows as well as in environments where you don't have permission to install libraries.

## Basic Usage
To get started with SDGym on Docker, you should do the following:

1. Pull the [latest SDGym image](https://hub.docker.com/r/sdvproject/sdgym) from DockerHub by running:

```bash
docker pull sdvproject/sdgym
```

2. Run SDGym using the following command:

```bash
docker run -ti sdvproject/sdgym -- sdgym COMMAND OPTIONS
```

where `COMMAND` and `OPTIONS` are the standard command line options. For example, you could try:

```bash
docker run -ti sdvproject/sdgym -- sdgym run --datasets adult --synthesizer sdv.tabular.CTGANSynthesizer
```

to benchmark the CTGAN model on the adult dataset.

## Advanced Usage
The above basic usage describes how you can run the benchmark on the built-in datasets which will 
automatically be downloaded inside your Docker container. However, if you already have a set of 
datasets stored locally that you want to use for benchmarking, you will need to mount the datasets
folder so it can be accessed from inside the Docker container:

```bash
docker run -ti -v </path/to/data>:/SDGym/datasets sdvproject/sdgym -- sdgym run --datasets-path /SDGym/datasets OPTIONS
```

The above command will take the path  `/path/to/data` and mount it inside the Docker container so
it is available to SDGym.
