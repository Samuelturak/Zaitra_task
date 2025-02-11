from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="s2_dataset_processor",  
    version="1.0",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Samuel Turák",
    author_email="samuel.turak@gmail.com",
    license="MIT",
    description="This package preprocesses the S2 dataset",
    url="https://github.com/Samuelturak",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
    ],
)
