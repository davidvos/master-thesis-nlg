### Introduction

**Introduce the general area of the research**

My thesis will be written on the domain of Natural Language Generation (NLG). Specifically, on data-to-text generation in (Dutch) news. In data-to-text, a language model produces a sequence of words based on a table of structured data. My thesis is written as part of an internship at DPG Media. For this reason, I want to give special attention to the Dutch news domain.

The current state-of-the-art for data-to-text is achieved by utilizing a large pre-trained language model (LM). The structural input is linearized and fed to the network as a sequence of tokens. The language model can be fine-tuned on task-specific data to obtain the highest possible performance.

The most recent examples of pre-trained language models are GPT-2, BART and T5. All of these are Transformer-based models. As Google's T5 achieves the current state-of-the-art performance on most language generation tasks, our research will focus on this pre-trained model. Following the release of T5, researchers have trained a multilingual version of this model. This will allow for research in English as well as in Dutch.

A current focus of the field is to capture structural data in a more efficient way than linearization. One line of research employs graph convolutional networks (GCNs) to do this. Another problem with large pre-trained LMs is that it is necessary to finetune billions of parameters for a single task. The resulting parameter space is often very close to the original one, suggesting that more efficient ways of fine-tuning are possible.

### Problem Statement

**What is the fundamental research problem that you aim to solve?**
 
A proposed way of fine-tuning is through learning prefixes. This method only learns about 0.1% of the parameters necessary for fine-tuning an entire language model. 

A language model takes in a series of input IDs. Each ID stands for a token that is present in the input sequence. The model then encodes this vector of IDs into an embedding that represents meaning in the context of the input sequence. Prefixes are continuous embedding vectors that are virtual 'tokens' prepended before the actual input tokens are processed. These prefixes can be conditioned on a single dataset, or more finegrained on input (meta)data and are learned through updating a single weight matrix while executing a generation task. The parameters of the pre-trained LM are all frozen.

Alhough prefix-tuning has shown to achieve SOTA performance on English data-to-text tasks, it has received relatively little attention. The main goal of this thesis is to evaluate prefix-tuning in alternative settings and experiment with the technique to achieve higher performance.

**Why is the problem important?**

Natural language modelling has shown great promise on a wide range of tasks. However, a single model already has billions of parameters. When fine-tuning for individual tasks, billions of new parameters are introduced for every new task. A 'new task' can also be interpreted as learning a task for a single user, making it important for finetuning approaches to become more scalable. Especially when deployed in a production environment. Prefix-tuning is extremely scalable and should thus be preferred when performance doesn't suffer from the limited expressiveness.

**Why is the problem non-trivial?**

Prefix tuning reduces 

**Why is the problem not solved by the current state-of-the-art**

It is using template based methods. Not using end-to-end.


### Proposed Approach & Contributions

**Sketch the method that you aim to develop to solve the outlined problem**
My thesis will focus on experimenting with different applications and approaches of prefix-tuning. These are the following:

- **The performance of prefix-tuning in a multilingual setting**
   Prefix-tuning has shown excellent performance on language generation tasks. For example, the WebNLG challenge is an English benchmark for data-to-text generation. However, we ask the question if the limited amount of prefix parameters contains enough expressiveness when the pre-trained LM is multilingual?

- **Conditioning prefix on structural information**
   Can we condition our prefix on information obtained my looking at functional relations in the input data?

**List the novel contributions provided by your approach**

Prefix-tuning is a novel approached that has only been explored to a certain amount. I will e

### Example

**Create an example, ideally with a figure, which exemplifies the problem and can be used to guide readers through your proposed approach**
 
Dutch sports reporting for amateur matches.

### Experimental Evaluation
 - How should the quality of your method be measured?
 - What are baseline techniques that your method should be compared to?
 - Sketch potential experiments and expected outcomes (on a high-level)

### Related Work

**List the most important related work, ideally with a paper reference and a one-sentence description**

### Open Questions

**List open questions not covered by the proposal**

### Next Steps

**Describe what the immediate next steps for the proposed work should be**