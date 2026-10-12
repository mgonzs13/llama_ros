FROM ubuntu:24.04

ARG ROS_DISTRO=jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV HEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0
ENV HEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      crossbuild-essential-arm64 \
      curl \
      git \
      lld \
      ninja-build \
      python3-empy \
      python3-lark \
      python3-numpy \
      python3-yaml \
      qemu-user-static \
      rsync \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      ros-${ROS_DISTRO}-ros-base \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/snapdragon-toolchain/arm64-linux:v0.7 /opt/hexagon /opt/hexagon

RUN mkdir -p /sysroot

WORKDIR /workspace

SHELL ["/bin/bash", "-c"]
