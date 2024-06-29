# llama_ros

This repository provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into ROS 2. By using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects by running [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)-based [LLMs](https://huggingface.co/models?sort=trending&search=gguf+7b) and [VLMs](https://huggingface.co/models?sort=trending&search=gguf+llava).

## Table of Contents

1. [Related Projects](#related-projects)
2. [Installation](#installation)
   - [CUDA](#cuda)
3. [Usage](#usage)
   - [Launch Files](#launch-files)
   - [ROS 2 Clients](#ros-2-clients)
   - [LangChain](#langchain)
4. [Demos](#demos)

## Related Projects

- [chatbot_ros](https://github.com/mgonzs13/chatbot_ros) &rarr; This chatbot, integrated into ROS 2, uses [whisper_ros](https://github.com/mgonzs13/whisper_ros/tree/main), to listen to people speech; and llama_ros, to generate responses. The chatbot is controlled by a state machine created with [YASMIN](https://github.com/uleroboticsgroup/yasmin).
- [explainable_ros](https://github.com/Dsobh/explainable_ROS) &rarr; A ROS 2 tool to explain the behavior of a robot. Using the integration of LangChain, logs are stored in a vector database. Then, RAG is applied to retrieve relevant logs for user questions that are answered with llama_ros.

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone --recurse-submodules https://github.com/mgonzs13/llama_ros.git
$ pip3 install -r llama_ros/requirements.txt
$ cd ~/ros2_ws
$ colcon build
```

### CUDA

To run llama_ros with CUDA, you have to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and set the environment variable `LLAMA_CUDA` to `on`:

```shell
export LLAMA_CUDA="on"
```

## Usage

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

            system_prompt_type="alpaca" # system prompt type
        )
    ])
```

```shell
$ ros2 launch llama_bringup marcoroni.launch.py
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

system_prompt_type: "ChatML" # system prompt type
```

```python
def generate_launch_description():
    return LaunchDescription([
        create_llama_launch_from_yaml(os.path.join(
            get_package_share_directory("llama_bringup"),
            "params", "Spaetzle-v60-7b-GGUF.yaml"))
    ])
```

```shell
$ ros2 launch llama_bringup marcoroni.launch.py
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
            embedding=False, # disable embeddings

            n_ctx=8192, # context of the LLM in tokens, use a huge context size to load images
            n_batch=512, # batch size in tokens
            n_gpu_layers=33, # layers to load in GPU
            n_threads=1, # threads
            n_predict=8192, # max tokens, -1 == inf

            model_repo="cjpais/llava-1.6-mistral-7b-gguf", # Hugging Face repo
            model_filename="llava-v1.6-mistral-7b.Q4_K_M.gguf", # model file in repo

            mmproj_repo="cjpais/llava-1.6-mistral-7b-gguf", # Hugging Face repo
            mmproj_filename="mmproj-model-f16.gguf", # mmproj file in repo

            system_prompt_type="mistral" # system prompt type
        )
    ])
```

```shell
$ ros2 launch llama_bringup llava.launch.py
```

</details>

#### llava_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
use_llava: True # enable llava
embedding: False # disable embeddings

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
            "params", "llava-1.6-mistral-7b-gguf.yaml"))
    ])
```

```shell
$ ros2 launch llama_bringup llava.launch.py
```

</details>

### ROS 2 Clients

Both llama_ros and llava_ros provide ROS 2 interfaces to access the main functionalities of the models. Here you have some examples of how to use them inside ROS 2 nodes. Moreover, take a look to the [llama_client_node.py](llama_ros/llama_ros/llama_client_node.py) and [llava_client_node.py](llama_ros/llama_ros/llava_client_node.py) examples.

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
        req.prompt = "Example text"

        # call the tokenize service
        self.srv_client.wait_for_service()
        res = self.srv_client.call(req)
        tokens = res.tokens
```

</details>

#### Embeddings

<details>
<summary>Click to expand</summary>

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
        res = self.srv_client.call(req)
        embeddings = res.embeddings
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
chain = prompt | model | StrOutputParser()

# run the chain
text = llm_chain.invoke({"topic": "bears"})

rclpy.shutdown()
```

</details>

#### llama_ros_embeddings (RAG)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from langchain_community.vectorstores import Chroma
from llama_ros.langchain import LlamaROSEmbeddings


rclpy.init()

# create the llama_ros embeddings for lanchain
embeddings = LlamaROSEmbeddings()

# create a vector database and assign it
db = Chroma(embedding_function=embeddings)

# create the retriever
retriever = self.db.as_retriever(search_kwargs={"k": 5})

# add your texts
db.add_texts(texts=["your_texts"])

# retrieve documents
docuemnts = self.retriever.get_relevant_documents("your_query")

rclpy.shutdown()
```

</details>

## Demos

### llama_ros

```shell
$ ros2 launch llama_bringup marcoroni.launch.py
```

```shell
$ ros2 run llama_ros llama_demo_node --ros-args -p prompt:="your prompt"
```

<!-- https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4 -->

https://github.com/mgonzs13/llama_ros/assets/25979134/9311761b-d900-4e58-b9f8-11c8efefdac4

### llava_ros

```shell
$ ros2 launch llama_bringup llava.launch.py
```

```shell
$ ros2 run llama_ros llava_demo_node --ros-args -p prompt:="your prompt" -p image_url:="url of the image" -p use_image:="whether to send the image"
```

https://github.com/mgonzs13/llama_ros/assets/25979134/4a9ef92f-9099-41b4-8350-765336e3503c
