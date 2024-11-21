string architecture             # Describes what architecture this model implements
int32 quantization_version      # The version of the quantization format. Not required if the model is not quantized
int32 alignment                 # The global alignment to use

string name                     # The name of the model. This should be a human-readable name that can be used to identify the model. It should be unique within the community that the model is defined in
string author                   # The author of the model
string version                  # The version of the model
string organization             # The organization of the model

string basename                 # The base model name / architecture of the model
string finetune                 # What has the base model been optimized toward
string description              # Free-form description of the model including anything that isn't covered by the other fields
string quantized_by             # The name of the individual who quantized the model
string size_label               # Size class of the model, such as number of weights and experts

string license                  # License of the model
string license_name             # Human friendly license name
string license_link             # URL to the license

string url                      # URL to the model's homepage
string repo_url                 # URL to the model's repository such as a GitHub repo or HuggingFace repo
string doi                      # Digital Object Identifier (DOI)
string uuid                     # Universally unique identifier

string file_type                # An enumerated value describing the type of the majority of the tensors in the file