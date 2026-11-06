# ROS 2 Clients

[← Back to README](../README.md)

Both llama_ros and llava_ros provide ROS 2 interfaces to access the main functionalities of the models. Here you have some examples of how to use them inside ROS 2 nodes. Moreover, take a look to the [llama_demo_node.py](../llama_demos/llama_demos/llama_demo_node.py) and [llava_demo_node.py](../llama_demos/llama_demos/llava_demo_node.py) demos.

## Tokenize

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

## Detokenize

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

        # call the detokenize service
        self.srv_client.wait_for_service()
        text = self.srv_client.call(req).text
```

</details>

## Embeddings

<details>
<summary>Click to expand</summary>

_Remember to launch llama_ros with embedding set to true to be able of generating embeddings with your LLM._

```python
from rclpy.node import Node
from llama_msgs.srv import GenerateEmbeddings


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(GenerateEmbeddings, "/llama/generate_embeddings")

        # create the request
        req = GenerateEmbeddings.Request()
        req.prompt = "Example text"
        req.normalization = 2  # -1=none, 0=max abs int16, 1=taxicab, 2=euclidean, >2=p-norm

        # call the embedding service
        self.srv_client.wait_for_service()
        embeddings = self.srv_client.call(req).embeddings
```

</details>

## Multimodal Embeddings

<details>
<summary>Click to expand</summary>

Multimodal embedding models (such as [EmbeddingGemma 2](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)) map text, images, and audio into a single vector space. They require `llava_node` with a multimodal projector (`mmproj`) and embedding mode enabled.

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/EmbeddingGemma-2.yaml
```

The `generate_embeddings` service accepts `sensor_msgs/Image[] images` and `std_msgs/UInt8MultiArray[] audios`. The prompt must contain one `<__media__>` placeholder per media item, in order (images first, then audios):

```python
from cv_bridge import CvBridge
import cv2
from llama_msgs.srv import GenerateEmbeddings

image = cv2.imread("/path/to/image.jpg")

req = GenerateEmbeddings.Request()
req.prompt = "task: sentence similarity | query: a red square <__media__>"
req.images.append(CvBridge().cv2_to_imgmsg(image))

embeddings = llama_client.generate_embeddings(req).embeddings  # 768 floats
```

Notes:

- `context.n_parallel: 1` is required: pooled embeddings are only correct when the whole multimodal prompt is decoded as a single sequence in a single batch. The service rejects media otherwise.
- The whole prompt (text + media) must fit in `min(context.n_batch, context.n_ubatch)`. An image costs 280 tokens at the default vision budget.
- EmbeddingGemma 2 has an 8K context window; set `context.n_ctx: 8192`.
- The full native output dimension is returned (768 for EmbeddingGemma 2).
- Use the model's task instruction prefixes (`task: search result | query: ...`, `task: sentence similarity | query: ...`, `title: ... | text: ...`) for best quality. Prefixes apply to text only.

</details>

## Generate Response

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

## Generate Response (llava)

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
        goal.images.append(self.cv_bridge.cv2_to_imgmsg(image))

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

## Generate Chat Completions

<details>
<summary>Click to expand</summary>

The `GenerateChatCompletions` action provides an OpenAI-compatible chat completions interface with support for tool calling, reasoning, and streaming.

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GenerateChatCompletions
from llama_msgs.msg import ChatMessage


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.action_client = ActionClient(
            self, GenerateChatCompletions, "/llama/generate_chat_completions")

        # create the goal
        goal = GenerateChatCompletions.Goal()
        goal.messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is ROS 2?")
        ]
        goal.sampling_config.temp = 0.2
        goal.stream = True

        # wait for the server and send the goal
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal)

        # wait for the server
        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        # wait again and take the result
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result
```

</details>

## Get Metadata

<details>
<summary>Click to expand</summary>

```python
from rclpy.node import Node
from llama_msgs.srv import GetMetadata


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(GetMetadata, "/llama/get_metadata")

        # call the metadata service
        req = GetMetadata.Request()
        self.srv_client.wait_for_service()
        metadata = self.srv_client.call(req).metadata
```

The response contains the classic `general`, `model` and `tokenizer` blocks
with untruncated values, extended with the model's `tags`, `languages`,
`datasets` and `base_models`, the tokenizer `add_eos_token`,
`mask_token_id` and named `chat_templates`, plus two new blocks: `sampling`
(the model's recommended sampling defaults) and `decision` (`enabled`,
`type`, `max_head_tokens`, `temperature_names`/`temperatures`,
`systemone_template`). The `decision` block tells whether the node exposes
`evaluate_decisions` and which upstream type the model declares, e.g. `laya`,
`kev` or `openjev`; for a plain chat model without decision metadata it is
disabled and empty. The `get_metadata` service is available for every model
type (completion, embedding, reranking and decision).

</details>

## Rerank Documents

<details>
<summary>Click to expand</summary>

_Remember to launch llama_ros with reranking set to true._

```python
from rclpy.node import Node
from llama_msgs.srv import RerankDocuments


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(RerankDocuments, "/llama/rerank_documents")

        # create the request
        req = RerankDocuments.Request()
        req.query = "What is robotics?"
        req.documents = ["Robotics is a field of engineering.", "The weather is sunny."]

        # call the reranking service
        self.srv_client.wait_for_service()
        scores = self.srv_client.call(req).scores
```

</details>
