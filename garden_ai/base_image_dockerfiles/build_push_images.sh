#!/bin/sh

echo "Starting garden-ai/python-3.8-jupyter"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.8-jupyter -f Dockerfile_jupyter_3.8 ./
docker tag garden-ai/python-3.8-jupyter:latest gardenai/base:python-3.8-jupyter
docker push gardenai/base:python-3.8-jupyter
docker image rm gardenai/base:python-3.8-jupyter

echo "Starting garden-ai/python-3.9-jupyter"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.9-jupyter -f Dockerfile_jupyter_3.9 ./
docker tag garden-ai/python-3.9-jupyter:latest gardenai/base:python-3.9-jupyter
docker push gardenai/base:python-3.9-jupyter
docker image rm gardenai/base:python-3.9-jupyter

echo "Starting garden-ai/python-3.10-jupyter"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.10-jupyter -f Dockerfile_jupyter_3.10 ./
docker tag garden-ai/python-3.10-jupyter:latest gardenai/base:python-3.10-jupyter
docker push gardenai/base:python-3.10-jupyter
docker image rm gardenai/base:python-3.10-jupyter

echo "Starting garden-ai/python-3.11-jupyter"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.11-jupyter -f Dockerfile_jupyter_3.11 ./
docker tag garden-ai/python-3.11-jupyter:latest gardenai/base:python-3.11-jupyter
docker push gardenai/base:python-3.11-jupyter
docker image rm gardenai/base:python-3.11-jupyter



echo "Starting garden-ai/python-3.8-jupyter-sklearn"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.8-jupyter-sklearn -f Dockerfile_jupyter_sklearn_3.8 ./
docker tag garden-ai/python-3.8-jupyter-sklearn:latest gardenai/base:python-3.8-jupyter-sklearn
docker push gardenai/base:python-3.8-jupyter-sklearn
docker image rm gardenai/base:python-3.8-jupyter-sklearn

echo "Starting garden-ai/python-3.9-jupyter-sklearn"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.9-jupyter-sklearn -f Dockerfile_jupyter_sklearn_3.9 ./
docker tag garden-ai/python-3.9-jupyter-sklearn:latest gardenai/base:python-3.9-jupyter-sklearn
docker push gardenai/base:python-3.9-jupyter-sklearn
docker image rm gardenai/base:python-3.9-jupyter-sklearn

echo "Starting garden-ai/python-3.10-jupyter-sklearn"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.10-jupyter-sklearn -f Dockerfile_jupyter_sklearn_3.10 ./
docker tag garden-ai/python-3.10-jupyter-sklearn:latest gardenai/base:python-3.10-jupyter-sklearn
docker push gardenai/base:python-3.10-jupyter-sklearn
docker image rm gardenai/base:python-3.10-jupyter-sklearn

echo "Starting garden-ai/python-3.11-jupyter-sklearn"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.11-jupyter-sklearn -f Dockerfile_jupyter_sklearn_3.11 ./
docker tag garden-ai/python-3.11-jupyter-sklearn:latest gardenai/base:python-3.11-jupyter-sklearn
docker push gardenai/base:python-3.11-jupyter-sklearn
docker image rm gardenai/base:python-3.11-jupyter-sklearn



echo "Starting garden-ai/python-3.8-jupyter-tf"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.8-jupyter-tf -f Dockerfile_jupyter_tf_3.8 ./
docker tag garden-ai/python-3.8-jupyter-tf:latest gardenai/base:python-3.8-jupyter-tf
docker push gardenai/base:python-3.8-jupyter-tf
docker image rm gardenai/base:python-3.8-jupyter-tf

echo "Starting garden-ai/python-3.9-jupyter-tf"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.9-jupyter-tf -f Dockerfile_jupyter_tf_3.9 ./
docker tag garden-ai/python-3.9-jupyter-tf:latest gardenai/base:python-3.9-jupyter-tf
docker push gardenai/base:python-3.9-jupyter-tf
docker image rm gardenai/base:python-3.9-jupyter-tf

echo "Starting garden-ai/python-3.10-jupyter-tf"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.10-jupyter-tf -f Dockerfile_jupyter_tf_3.10 ./
docker tag garden-ai/python-3.10-jupyter-tf:latest gardenai/base:python-3.10-jupyter-tf
docker push gardenai/base:python-3.10-jupyter-tf
docker image rm gardenai/base:python-3.10-jupyter-tf

echo "Starting garden-ai/python-3.11-jupyter-tf"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.11-jupyter-tf -f Dockerfile_jupyter_tf_3.11 ./
docker tag garden-ai/python-3.11-jupyter-tf:latest gardenai/base:python-3.11-jupyter-tf
docker push gardenai/base:python-3.11-jupyter-tf
docker image rm gardenai/base:python-3.11-jupyter-tf



echo "Starting garden-ai/python-3.8-jupyter-torch"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.8-jupyter-torch -f Dockerfile_jupyter_torch_3.8 ./
docker tag garden-ai/python-3.8-jupyter-torch:latest gardenai/base:python-3.8-jupyter-torch
docker push gardenai/base:python-3.8-jupyter-torch
docker image rm gardenai/base:python-3.8-jupyter-torch

echo "Starting garden-ai/python-3.9-jupyter-torch"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.9-jupyter-torch -f Dockerfile_jupyter_torch_3.9 ./
docker tag garden-ai/python-3.9-jupyter-torch:latest gardenai/base:python-3.9-jupyter-torch
docker push gardenai/base:python-3.9-jupyter-torch
docker image rm gardenai/base:python-3.9-jupyter-torch

echo "Starting garden-ai/python-3.10-jupyter-torch"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.10-jupyter-torch -f Dockerfile_jupyter_torch_3.10 ./
docker tag garden-ai/python-3.10-jupyter-torch:latest gardenai/base:python-3.10-jupyter-torch
docker push gardenai/base:python-3.10-jupyter-torch
docker image rm gardenai/base:python-3.10-jupyter-torch

echo "Starting garden-ai/python-3.11-jupyter-torch"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.11-jupyter-torch -f Dockerfile_jupyter_torch_3.11 ./
docker tag garden-ai/python-3.11-jupyter-torch:latest gardenai/base:python-3.11-jupyter-torch
docker push gardenai/base:python-3.11-jupyter-torch
docker image rm gardenai/base:python-3.11-jupyter-torch



echo "Starting garden-ai/python-3.10-jupyter-all"
docker buildx build --platform=linux/x86_64 -t garden-ai/python-3.10-jupyter-all -f Dockerfile_jupyter_all_3.10 ./
docker tag garden-ai/python-3.10-jupyter-all:latest gardenai/base:python-3.10-jupyter-all
docker push gardenai/base:python-3.10-jupyter-all
docker image rm gardenai/base:python-3.10-jupyter-all
