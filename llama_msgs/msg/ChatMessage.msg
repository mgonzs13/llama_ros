string role                         # The role of the message (user, system, etc.)
string content                      # The content of the message
ChatContent[] content_parts         # The content of the message if it is split into parts
ChatToolCall[] tool_calls           # The tool calls
string reasoning_content            # The reasoning content
string tool_name                    # The name of the required tool 
string tool_call_id                 # The ID of the tool call