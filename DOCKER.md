The DockerHub page can be found here: https://hub.docker.com/r/sdvproject/sdgym

To pull the SDGym Docker image, use the following command: `docker pull sdvproject/sdgym:<tag>`

After accessing the docker, run the following command to run SDGym: `docker run -ti sdvproject/sdgym:<tag> -- sdgym COMMAND OPTIONS.`

A section pointing out how to use a local collection of datasets by mounting a datasets folder: `docker run -ti -v <local-datasets-path>:/SDGym/datasets sdvproject/sdgym:<tag> -- sdgym run --datasets-path /SDGym/datasets OPTIONS`