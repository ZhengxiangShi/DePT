---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- superglue-multirc
metrics:
- f1
model-index:
- name: superglue-multirc_lr3e-1_loralr5e-3_pl60_r30_st30000_ml348
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# superglue-multirc_lr3e-1_loralr5e-3_pl60_r30_st30000_ml348

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on the superglue-multirc dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2119
- F1: 73.9130

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.3
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 30000

### Training results

| Training Loss | Epoch | Step  | Validation Loss | F1      |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|
| 0.3018        | 1.17  | 1000  | 0.2726          | 60.0540 |
| 0.2869        | 2.35  | 2000  | 0.2610          | 40.3731 |
| 0.2735        | 3.52  | 3000  | 0.2321          | 62.5199 |
| 0.2361        | 4.69  | 4000  | 0.2030          | 72.4432 |
| 0.2154        | 5.87  | 5000  | 0.2001          | 73.3083 |
| 0.2024        | 7.04  | 6000  | 0.2119          | 73.7185 |
| 0.2005        | 8.22  | 7000  | 0.2000          | 72.8261 |
| 0.1921        | 9.39  | 8000  | 0.2101          | 73.8550 |
| 0.1872        | 10.56 | 9000  | 0.2030          | 73.5051 |
| 0.1856        | 11.74 | 10000 | 0.1990          | 72.5549 |
| 0.1796        | 12.91 | 11000 | 0.2017          | 73.0034 |
| 0.1802        | 14.08 | 12000 | 0.2118          | 72.6014 |
| 0.178         | 15.26 | 13000 | 0.2111          | 72.2166 |
| 0.1798        | 16.43 | 14000 | 0.2060          | 73.1898 |
| 0.1752        | 17.61 | 15000 | 0.2193          | 73.7993 |
| 0.1724        | 18.78 | 16000 | 0.2106          | 73.4911 |
| 0.1731        | 19.95 | 17000 | 0.2002          | 73.4436 |
| 0.1735        | 21.13 | 18000 | 0.2080          | 73.3110 |
| 0.1696        | 22.3  | 19000 | 0.2184          | 73.3813 |
| 0.1676        | 23.47 | 20000 | 0.2147          | 73.2877 |
| 0.1697        | 24.65 | 21000 | 0.2190          | 73.0862 |
| 0.171         | 25.82 | 22000 | 0.2119          | 73.9130 |
| 0.1688        | 27.0  | 23000 | 0.2135          | 73.5051 |
| 0.1681        | 28.17 | 24000 | 0.2156          | 73.6538 |
| 0.1638        | 29.34 | 25000 | 0.2242          | 73.5266 |
| 0.1664        | 30.52 | 26000 | 0.2189          | 73.7705 |
| 0.1655        | 31.69 | 27000 | 0.2168          | 73.2326 |
| 0.1685        | 32.86 | 28000 | 0.2143          | 73.0205 |
| 0.166         | 34.04 | 29000 | 0.2156          | 73.7146 |
| 0.1668        | 35.21 | 30000 | 0.2151          | 73.5266 |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.13.1
- Datasets 2.10.1
- Tokenizers 0.13.2
