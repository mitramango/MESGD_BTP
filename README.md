<!-- omit in toc -->
# MESGD: Model Editing with Stochastic Gradient Descent

This repo contains the source code of our proposed MESGD, a model editing method based on the SGD and Adam optimizers.
<!-- omit in toc -->

<!-- omit in toc -->
## Table of Contents
- [Reference](#Reference)
- [Introduction](#introduction)
- [Experiments](#experiments)
- [Prepare Environments](#prepare-environments)
- [Prepare Datasets](#prepare-datasets)
- [Quick Start](#quick-start)
- [Acknowledgments](#Acknowledgments)

## Reference
We would appreciate if you could refer to our work as one of your baselines!
```
@article{yu2023melo,
  title={MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA},
  author={Yu, Lang and Chen, Qin and Zhou, Jie and He, Liang},
  journal={arXiv preprint arXiv:2312.11795},
  year={2023}
}
```
## Introduction
Due to the limitation of catastrophic forgetting and the lack of locality, few studies explore recent advanced Low-rank Adapter (LoRA) techniques for continual model editing. To overcome these limitations and take advantage of LoRA's resource efficiency, we propose MELO, a plug-in model editing method implemented with dynamic LoRA, which routes the behavior of language models by dynamically indexing LoRA blocks according to an inner vector database. MELO considers all editing properties and can be easily integrated into multiple LLMs such as BERT, T5 and GPT. Experimental results show that our proposed MELO achieves state-of-the-art editing performance on three sequential editing tasks (document classification, question answering and hallucination correction), while requires the least trainable parameters and computational cost.
![main](./figures/main_00.png)

## Experiments
Comparison of MELO to prior editing methods on sequential editing tasks. Note that MELO edits all language models with a single RTX 3090 GPU.
![table](./figures/table.png)

## Prepare Environments

Run the setup.sh file to setup the required environments.

Required CUDA environment and library dependencies are listed in: 
```
requirements.txt
```
Then you should install our modified PEFT:
<h1 align="center"> <p>🤗 PEFT-MELO</p></h1>

```
cd peft_egg
pip install -e .
```
Detailed implementation of MELO is in `./peft_egg/src/tuners/melo.py`
## Prepare Datasets
The zsRE experiments use data linked by the [MEND](https://github.com/eric-mitchell/mend) repository. Run the ``` download.sh ``` script to download and unzip the data into the correct directories.

### Editing T5 on zsRE with MESGD
```
cd btp
python run.py +alg=lora +experiment=qa +model=t5small
```

## Acknowledgments
We would like to thank the following individuals and organizations for their contributions to this project:
```
Huggingface: for their support of the PEFT community and their development of the PEFT framework (https://github.com/huggingface/peft)
```
