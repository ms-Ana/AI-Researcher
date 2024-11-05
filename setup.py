from setuptools import setup, find_packages

setup(
    name="ai_researcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic==0.34.1",
        "datasets==2.18.0",
        "matplotlib==3.7.4",
        "nltk==3.8.1",
        "numpy==1.22.4",
        "openai==1.42.0",
        "pandas==2.0.3",
        "Requests==2.32.3",
        "retry==0.9.2",
        "sentence_transformers==3.0.1",
        "tqdm==4.66.1",
    ],
    author=".",
    author_email=".",
    description="AI-researcher package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/my_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
