from setuptools import setup
from setuptools import find_packages

setup(
    name="llama_cli",
    version="4.1.4",
    packages=find_packages(exclude=["test"]),
    zip_safe=True,
    author="Miguel Ángel González Santamarta",
    author_email="mgons@unileon.es",
    maintainer="Miguel Ángel González Santamarta",
    maintainer_email="mgons@unileon.es",
    description="Cli package for llama_ros",
    license="MIT",
    data_files=[
        ("share/llama_cli", ["package.xml"]),
        ("share/ament_index/resource_index/packages", ["resource/llama_cli"]),
    ],
    entry_points={
        "ros2cli.command": [
            "llama = llama_cli.command.llama:LlamaCommand",
        ],
        "llama_cli.verb": [
            "launch = llama_cli.verb.launch:LaunchVerb",
            "prompt = llama_cli.verb.prompt:PromptVerb",
        ],
    },
)
