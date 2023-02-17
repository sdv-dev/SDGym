# Benchmark
This directory contains the standard benchmark script and Docker container which
is triggered monthly on AWS. To rebuild the container, run:

> docker build -t sdgym-benchmark .

To run the container manually, you can use:

> docker run -it --rm sdgym-benchmark --local

By specifying `--local`, the result will be printed instead of uploaded to S3.

To upload the container to AWS:

> TBD

The end.

## Infrastructure
This is designed to run on AWS Batch. The default compute environment will be an
`c7g.medium` machine which - assuming it takes a day to run all the benchmarks - 
should only cost a dollar.

We'll upload a docker container to AWS Elastic Container Registry and create a Job
Definition which takes the container and runs it. Finally, we'll configure Cloudwatch
to run this job once every month.
