ARG ROS_DISTRO=rolling
FROM ros:${ROS_DISTRO} AS deps

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
SHELL ["/bin/bash", "-c"]
COPY . /root/ros2_ws/src

# Install dependencies
RUN apt-get update \
    && apt-get -y --quiet --no-install-recommends install \
    gcc \
    git \
    wget \
    python3 \
    python3-pip

# Clone behavior_tree if ROS_DISTRO is rolling
RUN if [ "$ROS_DISTRO" = "rolling" ]; then \
    git clone https://github.com/BehaviorTree/BehaviorTree.CPP src/BehaviorTree.CPP; \
    fi

# Install rosdep
RUN apt update && rosdep install --from-paths src --ignore-src -r -y

# Check if ubuntu version is 24.04 or later
RUN if [ "$(lsb_release -rs)" = "24.04" ] || [ "$(lsb_release -rs)" = "24.10" ]; then \
    pip3 install -r src/requirements.txt --break-system-packages --ignore-installed; \
    else \
    pip3 install -r src/requirements.txt; \
    fi

# Install CUDA nvcc
ARG USE_CUDA
ARG CUDA_VERSION=12-6

RUN if [ "$USE_CUDA" = "1" ]; then \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb; \
    apt-get update && apt-get install -y cuda-toolkit-$CUDA_VERSION; \
    echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> ~/.bashrc; \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc; \
    fi

# Colcon the ws
FROM deps AS builder
ARG CMAKE_BUILD_TYPE=Release

ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    if [ "$USE_CUDA" = "1" ]; then \
    source ~/.bashrc && \
    colcon build --cmake-args -DGGML_CUDA=ON; \
    else \
    colcon build; \
    fi

# Source the ROS 2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Run a default command, e.g., starting a bash shell
CMD ["bash"]
