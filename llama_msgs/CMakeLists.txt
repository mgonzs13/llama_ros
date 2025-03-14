cmake_minimum_required(VERSION 3.8)
project(llama_msgs)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TokenProb.msg"
  "msg/TokenProbArray.msg"
  "msg/Response.msg"
  "msg/PartialResponse.msg"
  "msg/LogitBias.msg"
  "msg/LogitBiasArray.msg"
  "msg/SamplingConfig.msg"
  "msg/LoRA.msg"
  "msg/GeneralInfo.msg"
  "msg/TokenizerInfo.msg"
  "msg/AttentionInfo.msg"
  "msg/RoPEInfo.msg"
  "msg/ModelInfo.msg"
  "msg/Metadata.msg"
  "msg/GrammarTrigger.msg"
  "msg/ChatMessage.msg"
  "msg/ChatTool.msg"
  "msg/ChatContent.msg"
  "msg/ChatToolCall.msg"
  "msg/ChatReqTool.msg"
  "msg/ChatChoice.msg"
  "msg/ChatChoiceChunk.msg"
  "msg/ChatDeltaChunk.msg"
  "msg/UsageStats.msg"

  "action/GenerateResponse.action"
  "action/GenerateChatCompletions.action"

  "srv/GenerateEmbeddings.srv"
  "srv/Tokenize.srv"
  "srv/Detokenize.srv"
  "srv/ListLoRAs.srv"
  "srv/UpdateLoRAs.srv"
  "srv/RerankDocuments.srv"
  "srv/GetMetadata.srv"
  DEPENDENCIES sensor_msgs
)

ament_package()
