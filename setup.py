from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


MAJOR, MINOR, MICRO = 0, 1, 0
__VERSION__ = "{}.{}.{}".format(MAJOR, MINOR, MICRO)

setup(
    name="dnn-tip",
    version=__VERSION__,
    description=(
        "A collection of DNN test input prioritizers,"
        "in particular neuron coverage and surprise adequacy."
    ),
    long_description_content_type="text/markdown",
    long_description=readme(),
    # keywords="",
    url="https://github.com/testingautomated-usi/dnn-tip",
    author="Michael Weiss",
    author_email="michael.weiss@usi.ch",
    license="MIT",
    packages=["dnn_tip"],
    install_requires=["psutil", "scikit-learn", "tqdm"],
    extras_require={
        "lint": [
            "flake8==3.8.2",
            "black==22.3.0",
            "isort==5.6.4",
            "docstr-coverage==2.2.0",
        ],
        "test": ["pytest>=6.2.5"],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
