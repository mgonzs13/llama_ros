
from setuptools import setup
from setuptools import find_packages

setup(
    name="llama_cli",
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    install_requires=["ros2cli"],
    zip_safe=True,
    author="Miguel Ángel González Santamarta",
    author_email="mgons@unileon.es",
    maintainer="Miguel Ángel González Santamarta",
    maintainer_email="mgons@unileon.es",
    description="Cli package for llama_ros",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "ros2cli.command": [
            "llama = llama_cli.command.llama:LlamaCommand",
        ],
        "llama_cli.verb": [
            "launch = llama_cli.verb.launch:LaunchVerb",
        ]
    }
)
