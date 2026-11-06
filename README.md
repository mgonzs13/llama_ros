# llama_ros

This repository provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into ROS 2. Using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects by running [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)-based [LLMs](https://huggingface.co/models?sort=trending&search=gguf+7b), [VLMs](https://huggingface.co/models?sort=trending&search=gguf+llava), audio-language, embedding and reranking models. It also provides an OpenAI-compatible chat completions interface with tool calling, reasoning and streaming, as well as [speculative decoding](https://arxiv.org/abs/2302.01318) and decision models such as OpenJEV, Lev, Kev, Nimble, Laya and Clef. You can also use features from llama.cpp such as [GBNF grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) and modify LoRAs in real-time.

<div align="center">

[![License: MIT](https://img.shields.io/badge/GitHub-MIT-informational)](https://opensource.org/license/mit) [![GitHub release](https://img.shields.io/github/release/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/releases) [![Code Size](https://img.shields.io/github/languages/code-size/mgonzs13/llama_ros.svg?branch=main)](https://github.com/mgonzs13/llama_ros?branch=main) [![Last Commit](https://img.shields.io/github/last-commit/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/commits/main) [![GitHub issues](https://img.shields.io/github/issues/mgonzs13/llama_ros)](https://github.com/mgonzs13/llama_ros/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/mgonzs13/llama_ros)](https://github.com/mgonzs13/llama_ros/pulls) [![Contributors](https://img.shields.io/github/contributors/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/graphs/contributors) [![Python Formatter Check](https://github.com/mgonzs13/llama_ros/actions/workflows/python-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/python-formatter.yml?branch=main) [![C++ Formatter Check](https://github.com/mgonzs13/llama_ros/actions/workflows/cpp-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/cpp-formatter.yml?branch=main) [![Doxygen Deployment](https://github.com/mgonzs13/llama_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/llama_ros/latest)

| ROS 2 Distro |                          Branch                           |                                                                                                     Build status                                                                                                     |                                                                Docker Image                                                                |
| :----------: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
|  **Humble**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |  [![Humble Build](https://github.com/mgonzs13/llama_ros/actions/workflows/humble-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/humble-build-test.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-humble-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=humble)  |
|   **Iron**   | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |     [![Iron Build](https://github.com/mgonzs13/llama_ros/actions/workflows/iron-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/iron-build-test.yml?branch=main)      |    [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-iron-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=iron)    |
|  **Jazzy**   | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |    [![Jazzy Build](https://github.com/mgonzs13/llama_ros/actions/workflows/jazzy-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/jazzy-build-test.yml?branch=main)    |   [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-jazzy-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=jazzy)   |
|  **Kilted**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |  [![Kilted Build](https://github.com/mgonzs13/llama_ros/actions/workflows/kilted-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/kilted-build-test.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-kilted-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=kilted)  |
| **Lyrical**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) | [![Lyrical Build](https://github.com/mgonzs13/llama_ros/actions/workflows/lyrical-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/lyrical-build-test.yml?branch=main) | [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-lyrical-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=lyrical) |

</div>

## Quick Start

```shell
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/llama_ros.git
cd llama_ros
python3 -m pip install --upgrade pip || python3 -m pip install --break-system-packages --ignore-installed --upgrade pip
pip3 install --break-system-packages --ignore-installed -r requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args -DGGML_CUDA=ON  # omit this flag for a CPU-only build
source install/setup.bash
```

Launch a model and send a prompt:

```shell
ros2 llama launch $(ros2 pkg prefix llama_bringup)/share/llama_bringup/models/Llama-3.yaml
ros2 llama prompt "Do you know ROS 2?" -t 0.0
```

## Documentation

| Guide | Contents |
| ----- | -------- |
| [Installation](docs/installation.md) | Native installation, CUDA build and Docker |
| [Usage](docs/usage.md) | `llama_cli`, launch files and model configuration examples |
| [ROS 2 Parameters](docs/parameters.md) | Complete parameter reference with defaults |
| [ROS 2 Interfaces](docs/interfaces.md) | Messages, actions and services |
| [ROS 2 Clients](docs/clients.md) | Client examples for tokenization, generation, embeddings, reranking and metadata |
| [LangChain](docs/langchain.md) | LLM, RAG, chat, VLM, audio, tools, reasoning and agent examples |
| [Demos](docs/demos.md) | End-to-end demos for every supported model type |
| [Advanced Topics](docs/advanced.md) | LoRA adapters and links to advanced features |
| [Predefined Models](docs/models.md) | All `llama_bringup` model YAMLs |

## Related Projects

- [chatbot_ros](https://github.com/mgonzs13/chatbot_ros) &rarr; This chatbot, integrated into ROS 2, uses [whisper_ros](https://github.com/mgonzs13/whisper_ros/tree/main), to listen to people speech; and llama_ros, to generate responses. The chatbot is controlled by a state machine created with [YASMIN](https://github.com/uleroboticsgroup/yasmin).
- [explainable_ros](https://github.com/Dsobh/explainable_ROS) &rarr; A ROS 2 tool to explain the behavior of a robot. Using the integration of LangChain, logs are stored in a vector database. Then, RAG is applied to retrieve relevant logs for user questions answered with llama_ros.

## License

This project is licensed under the [MIT License](LICENSE).
