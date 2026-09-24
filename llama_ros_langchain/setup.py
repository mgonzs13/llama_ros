from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["test"]),
    zip_safe=True,
    tests_require=["pytest"],
    data_files=[
        ("share/llama_ros_langchain", ["package.xml"]),
        (
            "share/ament_index/resource_index/packages",
            ["resource/llama_ros_langchain"],
        ),
    ],
)
