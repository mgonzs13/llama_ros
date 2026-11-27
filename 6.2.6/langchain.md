# LangChain

[← Back to README](../README.md)

There is a [llama_ros_langchain](../llama_ros_langchain/llama_ros_langchain/) package, a llama_ros integration for LangChain. Thus, prompt engineering techniques could be applied. Here you have an example to use it.

## llama_ros (Chain)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros_langchain import LlamaROS
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

## llama_ros (Stream)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros_langchain import LlamaROS
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

## llava_ros

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros_langchain import LlamaROS

rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# bind the url_image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
llm = llm.bind(image_url=image_url).stream("Describe the image")

# run the llm
for c in llm:
    print(c, flush=True, end="")

rclpy.shutdown()
```

</details>

## llama_ros_embeddings (RAG)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from langchain_chroma import Chroma
from llama_ros_langchain import LlamaROSEmbeddings


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

## llama_ros (Reranker)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros_langchain import LlamaROSReranker
from llama_ros_langchain import LlamaROSEmbeddings

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

## llama_ros (LLM + RAG + Reranker)

<details>
<summary>Click to expand</summary>

```python
import bs4
import rclpy

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

from llama_ros_langchain import ChatLlamaROS, LlamaROSEmbeddings, LlamaROSReranker


rclpy.init()

# load, chunk and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=LlamaROSEmbeddings())

# retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are an AI assistant that answer questions briefly."),
        HumanMessagePromptTemplate.from_template(
            "Taking into account the followin information:{context}\n\n{question}"
        ),
    ]
)

# create rerank compression retriever
compressor = LlamaROSReranker(top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


def format_docs(docs):
    formated_docs = ""

    for d in docs:
        formated_docs += f"\n\n\t- {d.page_content}"

    return formated_docs


# create and use the chain
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatLlamaROS(temp=0.0)
    | StrOutputParser()
)

for c in rag_chain.stream("What is Task Decomposition?"):
    print(c, flush=True, end="")

rclpy.shutdown()
```

</details>

## chat_llama_ros (Chat + VLM)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros_langchain import ChatLlamaROS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create chat
chat = ChatLlamaROS(
    temp=0.2,
    penalty_last_n=8
)

# create prompt template with messages
prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a IA that just answer with a single word."),
    HumanMessagePromptTemplate.from_template(template=[
        {"type": "text", "text": "<__media__>Who is the character in the middle of the image?"},
        {"type": "image_url", "image_url": "{image_url}"}
    ])
])

# create the chain
chain = prompt | chat | StrOutputParser()

# stream and print the LLM output
for text in chain.stream({"image_url": "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg"}):
    print(text, end="", flush=True)

print("", end="\n", flush=True)

rclpy.shutdown()
```

</details>

## chat_llama_ros (Chat + Audio)

<details>
<summary>Click to expand</summary>

```python
import sys
import time
import rclpy
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_ros_langchain import ChatLlamaROS


def main():
    if len(sys.argv) < 2:
        prompt = "What's that sound?"
    else:
        prompt = " ".join(sys.argv[1:])

    tokens = 0
    initial_time = -1
    eval_time = -1

    rclpy.init()
    chat = ChatLlamaROS(temp=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("You are an IA that answer questions."),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": f"<__media__>{prompt}"},
                    {"type": "image_url", "image_url": "{audio_url}"},
                ]
            ),
        ]
    )

    chain = prompt | chat | StrOutputParser()

    initial_time = time.time()
    for text in chain.stream(
        {
            "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        }
    ):
        tokens += 1
        print(text, end="", flush=True)
        if eval_time < 0:
            eval_time = time.time()

    print("", end="\n", flush=True)

    end_time = time.time()
    print(f"Time to eval: {eval_time - initial_time} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time)} t/s")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

</details>

## chat_llama_ros (Structured output)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from llama_ros_langchain import ChatLlamaROS
from pydantic import BaseModel, Field

rclpy.init()

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

chat = ChatLlamaROS(temp=0.6, penalty_last_n=8)

structured_chat = chat.with_structured_output(
    Joke, method="function_calling"
)

prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "text", "text": "{prompt}"},
            ]
        ),
    ]
)

chain = prompt | structured_chat

res = chain.invoke({"prompt": "Tell me a joke about cats"})

print(f"Response: {res}")

rclpy.shutdown()
```

</details>

## chat_llama_ros (Tools)

<details>
<summary>Click to expand</summary>

The current implementation of Tools allows executing tools without requiring a model trained for that task.

```python
from random import randint

import rclpy

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from llama_ros_langchain import ChatLlamaROS

rclpy.init()

@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)

chat = ChatLlamaROS(temp=0.6, penalty_last_n=8)

messages = [
    HumanMessage(
        "What is the current temperature in Madrid? And its inhabitants?"
    )
]

llm_tools = chat.bind_tools(
    [get_inhabitants, get_curr_temperature], tool_choice='any'
)

all_tools_res = llm_tools.invoke(messages)
messages.append(all_tools_res)

for tool in all_tools_res.tool_calls:
    selected_tool = {
        "get_inhabitants": get_inhabitants, "get_curr_temperature": get_curr_temperature
    }[tool['name']]

    tool_msg = selected_tool.invoke(tool)

    formatted_output = f"{tool['name']}({''.join(tool['args'].values())}) = {tool_msg.content}"

    tool_msg.additional_kwargs = {'args': tool['args']}
    messages.append(tool_msg)

res = llm_tools.invoke(messages)

print(f"Response: {res.content}")

rclpy.shutdown()
```

</details>

## chat_llama_ros (Reasoning)

<details>
<summary>Click to expand</summary>

A reasoning model is required, such as Deepseek R1

```python
import time
from random import randint

import rclpy

from langchain_core.messages import HumanMessage
from llama_ros_langchain import ChatLlamaROS

rclpy.init()

chat = ChatLlamaROS(temp=0.6, penalty_last_n=8)

messages = [
    HumanMessage(
        "Here we have a book, a laptop, 9 eggs and a nail. Please tell me how to stack them onto each other in a stable manner."
    )
]

res = chat.invoke(messages)

print(f"Response: {res.content.strip()}")
print(f"Reasoning: {res.additional_kwargs["reasoning_content"]}")

rclpy.shutdown()
```

</details>

## chat_llama_ros (Agent)

<details>
<summary>Click to expand</summary>

```python
import time
from random import randint

import rclpy

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from llama_ros_langchain import ChatLlamaROS

rclpy.init()

@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)

chat = ChatLlamaROS(temp=0.0)

agent_executor = create_agent(
    chat, [get_inhabitants, get_curr_temperature]
)

response = agent_executor.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the current temperature in Madrid? And its inhabitants?"
            )
        ]
    }
)

print(f"Response: {response['messages'][-1].content}")

rclpy.shutdown()
```

</details>
