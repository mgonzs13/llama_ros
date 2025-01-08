# llama_ros

This repository provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into ROS 2. Using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects by running [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)-based [LLMs](https://huggingface.co/models?sort=trending&search=gguf+7b) and [VLMs](https://huggingface.co/models?sort=trending&search=gguf+llava). You can also use features from llama.cpp such as [GBNF grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) and modify LoRAs in real-time.

<div align="center">

[![License: MIT](https://img.shields.io/badge/GitHub-MIT-informational)](https://opensource.org/license/mit) [![GitHub release](https://img.shields.io/github/release/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/releases) [![Code Size](https://img.shields.io/github/languages/code-size/mgonzs13/llama_ros.svg?branch=main)](https://github.com/mgonzs13/llama_ros?branch=main) [![Last Commit](https://img.shields.io/github/last-commit/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/commits/main) [![GitHub issues](https://img.shields.io/github/issues/mgonzs13/llama_ros)](https://github.com/mgonzs13/llama_ros/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/mgonzs13/llama_ros)](https://github.com/mgonzs13/llama_ros/pulls) [![Contributors](https://img.shields.io/github/contributors/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/graphs/contributors) [![Python Formatter Check](https://github.com/mgonzs13/llama_ros/actions/workflows/python-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/python-formatter.yml?branch=main) [![C++ Formatter Check](https://github.com/mgonzs13/llama_ros/actions/workflows/cpp-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/cpp-formatter.yml?branch=main)

| ROS 2 Distro |                          Branch                           |                                                                                                     Build status                                                                                                      |                                                               Docker Image                                                               | Documentation                                                                                                                                            |
| :----------: | :-------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **Humble**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) | [![Humble Build](https://github.com/mgonzs13/llama_ros/actions/workflows/humble-docker-build.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/humble-docker-build.yml?branch=main) | [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-humble-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=humble) | [![Doxygen Deployment](https://github.com/mgonzs13/llama_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/llama_ros/) |

</div>

## Table of Contents

1. [Related Projects](#related-projects)
2. [Installation](#installation)
3. [Docker](#docker)
4. [Usage](#usage)
   - [llama_cli](#llama_cli)
   - [Launch Files](#launch-files)
   - [LoRA Adapters](#lora-adapters)
   - [ROS 2 Clients](#ros-2-clients)
   - [LangChain](#langchain)
5. [Demos](#demos)

## Related Projects

- [chatbot_ros](https://github.com/mgonzs13/chatbot_ros) &rarr; This chatbot, integrated into ROS 2, uses [whisper_ros](https://github.com/mgonzs13/whisper_ros/tree/main), to listen to people speech; and llama_ros, to generate responses. The chatbot is controlled by a state machine created with [YASMIN](https://github.com/uleroboticsgroup/yasmin).
- [explainable_ros](https://github.com/Dsobh/explainable_ROS) &rarr; A ROS 2 tool to explain the behavior of a robot. Using the integration of LangChain, logs are stored in a vector database. Then, RAG is applied to retrieve relevant logs for user questions answered with llama_ros.

## Installation

To run llama_ros with CUDA, first, you must install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then, you can compile llama_ros with `--cmake-args -DGGML_CUDA=ON` to enable CUDA support.

```shell
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/llama_ros.git
pip3 install -r llama_ros/requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args -DGGML_CUDA=ON # add this for CUDA
```

## Docker

Build the llama_ros docker or download an image from [DockerHub](https://hub.docker.com/repository/docker/mgons/llama_ros). You can choose to build llama_ros with CUDA (`USE_CUDA`) and choose the CUDA version (`CUDA_VERSION`). Remember that you have to use `DOCKER_BUILDKIT=0` to compile llama_ros with CUDA when building the image.

<!-- To build using CUDA you have to install the [NVIDIA Container Tollkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [configure the default runtime to NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.12.1/user-guide.html#daemon-configuration-file). -->

```shell
DOCKER_BUILDKIT=0 docker build -t llama_ros --build-arg USE_CUDA=1 --build-arg CUDA_VERSION=12-6 .
```

Run the docker container. If you want to use CUDA, you have to install the [NVIDIA Container Tollkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and add `--gpus all`.

```shell
docker run -it --rm --gpus all llama_ros
```

## Usage

### llama_cli

Commands are included in llama_ros to speed up the test of GGUF-based LLMs within the ROS 2 ecosystem. This way, the following commands are integrating into the ROS 2 commands:

#### launch

Using this command launch a LLM from a YAML file. The configuration of the YAML is used to launch the LLM in the same way as using a regular launch file. Here is an example of how to use it:

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/StableLM-Zephyr.yaml
```

#### prompt

Using this command send a prompt to a launched LLM. The command uses a string, which is the prompt and has the following arguments:

- (`-r`, `--reset`): Whether to reset the LLM before prompting
- (`-t`, `--temp`): The temperature value
- (`--image-url`): Image url to sent to a VLM

Here is an example of how to use it:

```shell
ros2 llama prompt "Do you know ROS 2?" -t 0.0
```

### Launch Files

First of all, you need to create a launch file to use llama_ros or llava_ros. This launch file will contain the main parameters to download the model from HuggingFace and configure it. Take a look at the following examples and the [predefined launch files](llama_bringup/launch).

#### llama_ros (Python Launch)

<details>
<summary>Click to expand</summary>

```python
from launch import LaunchDescription
from llama_bringup.utils import create_llama_launch


def generate_launch_description():

    return LaunchDescription([
        create_llama_launch(
            n_ctx=2048, # context of the LLM in tokens
            n_batch=8, # batch size in tokens
            n_gpu_layers=0, # layers to load in GPU
            n_threads=1, # threads
            n_predict=2048, # max tokens, -1 == inf

            model_repo="TheBloke/Marcoroni-7B-v3-GGUF", # Hugging Face repo
            model_filename="marcoroni-7b-v3.Q4_K_M.gguf", # model file in repo

            system_prompt_type="Alpaca" # system prompt type
        )
    ])
```

```shell
ros2 launch llama_bringup marcoroni.launch.py
```

</details>

#### llama_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
n_ctx: 2048 # context of the LLM in tokens
n_batch: 8 # batch size in tokens
n_gpu_layers: 0 # layers to load in GPU
n_threads: 1 # threads
n_predict: 2048 # max tokens, -1 == inf

model_repo: "cstr/Spaetzle-v60-7b-GGUF" # Hugging Face repo
model_filename: "Spaetzle-v60-7b-q4-k-m.gguf" # model file in repo

system_prompt_type: "Alpaca" # system prompt type
```

```python
import os
from launch import LaunchDescription
from llama_bringup.utils import create_llama_launch_from_yaml
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        create_llama_launch_from_yaml(os.path.join(
            get_package_share_directory("llama_bringup"), "models", "Spaetzle.yaml"))
    ])
```

```shell
ros2 launch llama_bringup spaetzle.launch.py
```

</details>

#### llama_ros (YAML Config + model shards)

<details>
<summary>Click to expand</summary>

```yaml
n_ctx: 2048 # context of the LLM in tokens
n_batch: 8 # batch size in tokens
n_gpu_layers: 0 # layers to load in GPU
n_threads: 1 # threads
n_predict: 2048 # max tokens, -1 == inf

model_repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF" # Hugging Face repo
model_filename: "qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf" # model shard file in repo

system_prompt_type: "ChatML" # system prompt type
```

```shell
ros2 llama launch Qwen2.yaml
```

</details>

#### llava_ros (Python Launch)

<details>
<summary>Click to expand</summary>

```python
from launch import LaunchDescription
from llama_bringup.utils import create_llama_launch

def generate_launch_description():

    return LaunchDescription([
        create_llama_launch(
            use_llava=True, # enable llava

            n_ctx=8192, # context of the LLM in tokens, use a huge context size to load images
            n_batch=512, # batch size in tokens
            n_gpu_layers=33, # layers to load in GPU
            n_threads=1, # threads
            n_predict=8192, # max tokens, -1 == inf

            model_repo="cjpais/llava-1.6-mistral-7b-gguf", # Hugging Face repo
            model_filename="llava-v1.6-mistral-7b.Q4_K_M.gguf", # model file in repo

            mmproj_repo="cjpais/llava-1.6-mistral-7b-gguf", # Hugging Face repo
            mmproj_filename="mmproj-model-f16.gguf", # mmproj file in repo

            system_prompt_type="Mistral" # system prompt type
        )
    ])
```

```shell
ros2 launch llama_bringup llava.launch.py
```

</details>

#### llava_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
use_llava: True # enable llava

n_ctx: 8192 # context of the LLM in tokens use a huge context size to load images
n_batch: 512 # batch size in tokens
n_gpu_layers: 33 # layers to load in GPU
n_threads: 1 # threads
n_predict: 8192 # max tokens -1 : :  inf

model_repo: "cjpais/llava-1.6-mistral-7b-gguf" # Hugging Face repo
model_filename: "llava-v1.6-mistral-7b.Q4_K_M.gguf" # model file in repo

mmproj_repo: "cjpais/llava-1.6-mistral-7b-gguf" # Hugging Face repo
mmproj_filename: "mmproj-model-f16.gguf" # mmproj file in repo

system_prompt_type: "mistral" # system prompt type
```

```python
def generate_launch_description():
    return LaunchDescription([
        create_llama_launch_from_yaml(os.path.join(
            get_package_share_directory("llama_bringup"),
            "models", "llava-1.6-mistral-7b-gguf.yaml"))
    ])
```

```shell
ros2 launch llama_bringup llava.launch.py
```

</details>

### LoRA Adapters

You can use LoRA adapters when launching LLMs. Using llama.cpp features, you can load multiple adapters choosing the scale to apply for each adapter. Here you have an example of using LoRA adapters with Phi-3. You can lis the
LoRAs using the `/llama/list_loras` service and modify their scales values by using the `/llama/update_loras` service. A scale value of 0.0 means not using that LoRA.

<details>
<summary>Click to expand</summary>

```yaml
n_ctx: 2048
n_batch: 8
n_gpu_layers: 0
n_threads: 1
n_predict: 2048

model_repo: "bartowski/Phi-3.5-mini-instruct-GGUF"
model_filename: "Phi-3.5-mini-instruct-Q4_K_M.gguf"

lora_adapters:
  - repo: "zhhan/adapter-Phi-3-mini-4k-instruct_code_writing"
    filename: "Phi-3-mini-4k-instruct-adaptor-f16-code_writer.gguf"
    scale: 0.5
  - repo: "zhhan/adapter-Phi-3-mini-4k-instruct_summarization"
    filename: "Phi-3-mini-4k-instruct-adaptor-f16-summarization.gguf"
    scale: 0.5

system_prompt_type: "Phi-3"
```

</details>

### ROS 2 Clients

Both llama_ros and llava_ros provide ROS 2 interfaces to access the main functionalities of the models. Here you have some examples of how to use them inside ROS 2 nodes. Moreover, take a look to the [llama_demo_node.py](llama_demos/llama_demos/llama_demo_node.py) and [llava_demo_node.py](llama_demos/llama_demos/llava_demo_node.py) demos.

#### Tokenize

<details>
<summary>Click to expand</summary>

```python
from rclpy.node import Node
from llama_msgs.srv import Tokenize


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(Tokenize, "/llama/tokenize")

        # create the request
        req = Tokenize.Request()
        req.text = "Example text"

        # call the tokenize service
        self.srv_client.wait_for_service()
        tokens = self.srv_client.call(req).tokens
```

</details>

#### Detokenize

<details>
<summary>Click to expand</summary>

```python
from rclpy.node import Node
from llama_msgs.srv import Detokenize


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(Detokenize, "/llama/detokenize")

        # create the request
        req = Detokenize.Request()
        req.tokens = [123, 123]

        # call the tokenize service
        self.srv_client.wait_for_service()
        text = self.srv_client.call(req).text
```

</details>

#### Embeddings

<details>
<summary>Click to expand</summary>

_Remember to launch llama_ros with embedding set to true to be able of generating embeddings with your LLM._

```python
from rclpy.node import Node
from llama_msgs.srv import Embeddings


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(Embeddings, "/llama/generate_embeddings")

        # create the request
        req = Embeddings.Request()
        req.prompt = "Example text"
        req.normalize = True

        # call the embedding service
        self.srv_client.wait_for_service()
        embeddings = self.srv_client.call(req).embeddings
```

</details>

#### Generate Response

<details>
<summary>Click to expand</summary>

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GenerateResponse


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.action_client = ActionClient(
            self, GenerateResponse, "/llama/generate_response")

        # create the goal and set the sampling config
        goal = GenerateResponse.Goal()
        goal.prompt = self.prompt
        goal.sampling_config.temp = 0.2

        # wait for the server and send the goal
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(
            goal)

        # wait for the server
        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        # wait again and take the result
        rclpy.spin_until_future_complete(self, get_result_future)
        result: GenerateResponse.Result = get_result_future.result().result
```

</details>

#### Generate Response (llava)

<details>
<summary>Click to expand</summary>

```python
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GenerateResponse


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create a cv bridge for the image
        self.cv_bridge = CvBridge()

        # create the client
        self.action_client = ActionClient(
            self, GenerateResponse, "/llama/generate_response")

        # create the goal and set the sampling config
        goal = GenerateResponse.Goal()
        goal.prompt = self.prompt
        goal.sampling_config.temp = 0.2

        # add your image to the goal
        image = cv2.imread("/path/to/your/image", cv2.IMREAD_COLOR)
        goal.image = self.cv_bridge.cv2_to_imgmsg(image)

        # wait for the server and send the goal
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(
            goal)

        # wait for the server
        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        # wait again and take the result
        rclpy.spin_until_future_complete(self, get_result_future)
        result: GenerateResponse.Result = get_result_future.result().result
```

</details>

### LangChain

There is a [llama_ros integration for LangChain](llama_ros/llama_ros/langchain/). Thus, prompt engineering techniques could be applied. Here you have an example to use it.

#### llama_ros (Chain)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# create a prompt template
prompt_template = "tell me a joke about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=prompt_template
)

# create a chain with the llm and the prompt template
chain = prompt | llm | StrOutputParser()

# run the chain
text = chain.invoke({"topic": "bears"})
print(text)

rclpy.shutdown()
```

</details>

#### llama_ros (Stream)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# create a prompt template
prompt_template = "tell me a joke about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=prompt_template
)

# create a chain with the llm and the prompt template
chain = prompt | llm | StrOutputParser()

# run the chain
for c in chain.stream({"topic": "bears"}):
    print(c, flush=True, end="")

rclpy.shutdown()
```

</details>

### llava_ros

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROS

rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# bind the url_image
llm = llm.bind(image_url=image_url).stream("Describe the image")
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

# run the llm
for c in llm:
    print(c, flush=True, end="")

rclpy.shutdown()

```

</details>

#### llama_ros_embeddings (RAG)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from langchain_chroma import Chroma
from llama_ros.langchain import LlamaROSEmbeddings


rclpy.init()

# create the llama_ros embeddings for langchain
embeddings = LlamaROSEmbeddings()

# create a vector database and assign it
db = Chroma(embedding_function=embeddings)

# create the retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# add your texts
db.add_texts(texts=["your_texts"])

# retrieve documents
documents = retriever.invoke("your_query")
print(documents)

rclpy.shutdown()
```

</details>

#### llama_ros (Renranker)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROSReranker
from llama_ros.langchain import LlamaROSEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever


rclpy.init()

# load the documents
documents = TextLoader("../state_of_the_union.txt",).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# create the llama_ros embeddings
embeddings = LlamaROSEmbeddings()

# create the VD and the retriever
retriever = FAISS.from_documents(
    texts, embeddings).as_retriever(search_kwargs={"k": 20})

# create the compressor using the llama_ros reranker
compressor = LlamaROSReranker()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# retrieve the documents
compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)

for doc in compressed_docs:
    print("-" * 50)
    print(doc.page_content)
    print("\n")

rclpy.shutdown()
```

</details>

#### llama_ros (LLM + RAG + Reranker)

<details>
<summary>Click to expand</summary>

```python
import bs4
import rclpy
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_ros.langchain import LlamaROS, LlamaROSEmbeddings, LlamaROSReranker
from langchain.retrievers import ContextualCompressionRetriever


rclpy.init()

# load, chunk and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=LlamaROSEmbeddings())

# retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
prompt = hub.pull("rlm/rag-prompt")

compressor = LlamaROSReranker(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# create and use the chain
rag_chain = (
    {"context": compression_retriever | format_docs,
        "question": RunnablePassthrough()}
    | prompt
    | LlamaROS(temp=0.0)
    | StrOutputParser()
)

print(rag_chain.invoke("What is Task Decomposition?"))

rclpy.shutdown()
```

</details>

#### chat_llama_ros

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import ChatLlamaROS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create chat
chat = ChatLlamaROS(
    temp=0.2,
    penalty_last_n=8,
)

# create prompt template with messages
prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a IA that just answer with a single word."),
    HumanMessagePromptTemplate.from_template(template=[
        {"type": "text", "text": "<image>Who is the character in the middle of the image?"},
        {"type": "image_url", "image_url": "{image_url}"}
    ])
])

# create the chain
chain = prompt | chat | StrOutputParser()

# stream and print the LLM output
for text in self.chain.stream({"image_url": "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg"}):
    print(text, end="", flush=True)

print("", end="\n", flush=True)

rclpy.shutdown()
```

</details>

## Demos

### LLM Demo

```shell
ros2 launch llama_bringup spaetzle.launch.py
```

```shell
ros2 run llama_demos llama_demo_node --ros-args -p prompt:="your prompt"
```

<!-- https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4 -->

https://github.com/mgonzs13/llama_ros/assets/25979134/9311761b-d900-4e58-b9f8-11c8efefdac4

### Embeddings Generation Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/bge-base-en-v1.5.yaml
```

```shell
ros2 run llama_demos llama_embeddings_demo_node
```

https://github.com/user-attachments/assets/7d722017-27dc-417c-ace7-bf6b747e4ced

### Reranking Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/jina-reranker.yaml
```

```shell
ros2 run llama_demos llama_rerank_demo_node
```

https://github.com/user-attachments/assets/4b4adb4d-7c70-43ea-a2c1-9be57d211484

### VLM Demo

```shell
ros2 launch llama_bringup minicpm-2.6.launch.py
```

```shell
ros2 run llama_demos llava_demo_node --ros-args -p prompt:="your prompt" -p image_url:="url of the image" -p use_image:="whether to send the image"
```

https://github.com/mgonzs13/llama_ros/assets/25979134/4a9ef92f-9099-41b4-8350-765336e3503c

### Chat Template Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6</summary>

```yaml
use_llava: True

n_ctx: 8192
n_batch: 512
n_gpu_layers: 20
n_threads: 1
n_predict: 8192

image_prefix: "<image>"
image_suffix: "</image>"

model_repo: "openbmb/MiniCPM-V-2_6-gguf"
model_filename: "ggml-model-Q4_K_M.gguf"

mmproj_repo: "openbmb/MiniCPM-V-2_6-gguf"
mmproj_filename: "mmproj-model-f16.gguf"

stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_demo_node
```

[ChatLlamaROS demo](https://github-production-user-asset-6210df.s3.amazonaws.com/55236157/363094669-c6de124a-4e91-4479-99b6-685fecb0ac20.webm?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240830%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240830T081232Z&X-Amz-Expires=300&X-Amz-Signature=f937758f4bcbaec7683e46ddb057fb642dc86a33cc8c736fca3b5ce2bf06ddac&X-Amz-SignedHeaders=host&actor_id=55236157&key_id=0&repo_id=622137360)

#### Full Demo (LLM + chat template + RAG + Reranking + Stream)

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/bge-base-en-v1.5.yaml
```

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/jina-reranker.yaml
```

```shell
ros2 llama launch Llama-3.yaml
```

<details>
<summary>Click to expand Llama-3.yaml</summary>

```yaml
n_ctx: 4096
n_batch: 256
n_gpu_layers: 33
n_threads: -1
n_predict: -1

model_repo: "lmstudio-community/Llama-3.2-1B-Instruct-GGUF"
model_filename: "Llama-3.2-1B-Instruct-Q8_0.gguf"

stopping_words: ["<|eot_id|>"]
```

</details>

```shell
ros2 run llama_demos llama_rag_demo_node
```

https://github.com/user-attachments/assets/b4e3957d-1f92-427b-a1a8-cfc76737c0d6
