# Installation

[← Back to README](../README.md)

To run llama_ros with CUDA, first, you must install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then, you can compile llama_ros with `--cmake-args -DGGML_CUDA=ON` to enable CUDA support.

Then clone the repository and install the Python dependencies:

```shell
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/llama_ros.git
cd llama_ros
python3 -m pip install --upgrade pip || python3 -m pip install --break-system-packages --ignore-installed --upgrade pip
pip3 install --break-system-packages --ignore-installed -r requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args -DGGML_CUDA=ON # add this for CUDA
```

To run the tests:

```shell
colcon test --executor sequential --packages-select llama_ros llama_bt llama_cli
colcon test-result --verbose
```

## Docker

Build the llama_ros docker or download an image from [DockerHub](https://hub.docker.com/repository/docker/mgons/llama_ros). You can choose to build llama_ros with CUDA (`USE_CUDA`) and choose the CUDA version (`CUDA_VERSION`). Remember that you have to use `DOCKER_BUILDKIT=0` to compile llama_ros with CUDA when building the image.

<!-- To build using CUDA you have to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [configure the default runtime to NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.12.1/user-guide.html#daemon-configuration-file). -->

```shell
DOCKER_BUILDKIT=0 docker build -t llama_ros --build-arg USE_CUDA=1 --build-arg CUDA_VERSION=12-6 .
```

Run the docker container. If you want to use CUDA, you have to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and add `--gpus all`.

```shell
docker run -it --rm --gpus all llama_ros
```
