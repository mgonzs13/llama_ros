ARG ROS_DISTRO=rolling
FROM ros:${ROS_DISTRO} AS deps

SHELL ["/bin/bash", "-c"]

# Copy sources and import dependencies
WORKDIR /root/ros2_ws
COPY . src/llama_ros/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    git \
    wget \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

# Clone BehaviorTree.CPP if ROS_DISTRO is rolling
RUN if [ "$ROS_DISTRO" = "rolling" ]; then \
    git clone https://github.com/BehaviorTree/BehaviorTree.CPP src/BehaviorTree.CPP; \
    fi

# Install ROS dependencies
RUN apt-get update && \
    rosdep update --include-eol-distros && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with uv
RUN cd src/llama_ros && uv sync

# Install CUDA toolkit (optional)
ARG USE_CUDA=0
ARG CUDA_VERSION=12-6

RUN if [ "$USE_CUDA" = "1" ]; then \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-${CUDA_VERSION} && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# Build the workspace with colcon
FROM deps AS builder
ARG CMAKE_BUILD_TYPE=Release
ARG USE_CUDA=0

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    if [ "$USE_CUDA" = "1" ]; then \
        colcon build --cmake-args -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}; \
    else \
        colcon build --cmake-args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}; \
    fi

# Source the workspace and activate the uv environment on login
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/src/llama_ros/.venv/bin/activate" >> ~/.bashrc

CMD ["bash"]
