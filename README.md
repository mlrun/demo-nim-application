# **Banking AI Bot** - Multi-Agent System with NVIDIA NIMs

<img src="./images/mlrunXnvidia.png" alt="nim-mlrun" style="width: 800px"/>

This demo shows how to deploy NVIDIA NIM (NVIDIA Inference Microservices) and build an AI application: a **Multi-Agent Banking Bot**! We'll deploy [**NVIDIA's NIM**](https://developer.nvidia.com/nim) microservices and cover how easy it is to take them to production with monitoring, scaling and MLOps best practices. MLRun handles all the complexity!

We will use:
* [**NVIDIA NIM**](https://developer.nvidia.com/nim) - for GPU-accelerated model serving
* [**MLRun**](https://www.mlrun.org/) - as the orchestrator to operationalize it all
* [**LangChain**](https://www.langchain.com/) - as the main framework for building the AI logic

The demo contains a single [notebook](./mlrun-nim-demo.ipynb) covering two main stages:

* **Model Serving & Monitoring** - Deploy a NIM, add MLRun's LLM Gateway for modularity and monitoring capabilities
* **Application Pipeline** - Build a multi-agent banking chatbot using MLRun's GenAI Factory components

You can find all the python source code under [/src](./src)

___

## Installation

This project can run in different development environments:
* Local computer (using PyCharm, VSCode, Jupyter, etc.)
* Inside GitHub Codespaces
* Other managed Jupyter environments

### Install the code and dependencies
```bash
pip install mlrun langchain_nvidia_ai_endpoints langchain-openai
```

### Required credentials
* NVIDIA NGC API Key
* OpenAI API Key (for monitoring)

