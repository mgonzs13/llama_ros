from langchain_core.runnables import chain
from langchain_core.messages.base import BaseMessage

from llama_msgs.srv import ChatMessages
from llama_msgs.msg import Message
from llama_ros.llama_client_node import LlamaClientNode


@chain
def ChatPromptFormatter(messages):
    client = LlamaClientNode.get_instance('llama')
    output_msgs = []

    for msg in messages.messages:
        new_msg = Message()
        new_msg.role = msg.type
        new_msg.content = msg.content
        output_msgs.append(new_msg)
    
    response = client.format_chat_prompt(ChatMessages.Request(messages=output_msgs))

    return response.formatted_prompt