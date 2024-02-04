---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- superglue-cb
metrics:
- accuracy
model-index:
- name: superglue-cb_lr4e-1_loralr1e-4_pl60_r30_st30000
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# superglue-cb_lr4e-1_loralr1e-4_pl60_r30_st30000

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on the superglue-cb dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2389
- F1 Multiclass: 94.5887
- Accuracy: 92.8571

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.4
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 30000

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | F1 Multiclass | Accuracy |
|:-------------:|:------:|:-----:|:---------------:|:-------------:|:--------:|
| 0.308         | 125.0  | 1000  | 0.3121          | 52.9293       | 75.0     |
| 0.1064        | 250.0  | 2000  | 0.2914          | 75.8586       | 85.7143  |
| 0.0418        | 375.0  | 3000  | 0.4492          | 81.2903       | 82.1429  |
| 0.0206        | 500.0  | 4000  | 0.3994          | 86.8548       | 89.2857  |
| 0.0116        | 625.0  | 5000  | 0.5067          | 91.7898       | 89.2857  |
| 0.0063        | 750.0  | 6000  | 0.3503          | 91.7898       | 89.2857  |
| 0.0053        | 875.0  | 7000  | 0.5653          | 91.7898       | 89.2857  |
| 0.0028        | 1000.0 | 8000  | 0.6196          | 91.7898       | 89.2857  |
| 0.003         | 1125.0 | 9000  | 0.6441          | 84.1270       | 85.7143  |
| 0.0026        | 1250.0 | 10000 | 0.4909          | 91.7898       | 89.2857  |
| 0.0015        | 1375.0 | 11000 | 0.4583          | 91.7898       | 89.2857  |
| 0.0021        | 1500.0 | 12000 | 0.5380          | 91.7898       | 89.2857  |
| 0.0013        | 1625.0 | 13000 | 0.2258          | 91.7898       | 89.2857  |
| 0.0017        | 1750.0 | 14000 | 0.5343          | 91.7898       | 89.2857  |
| 0.0011        | 1875.0 | 15000 | 0.5411          | 91.7898       | 89.2857  |
| 0.0009        | 2000.0 | 16000 | 0.5722          | 91.7898       | 89.2857  |
| 0.0005        | 2125.0 | 17000 | 0.4313          | 91.7898       | 89.2857  |
| 0.0007        | 2250.0 | 18000 | 0.5533          | 91.7898       | 89.2857  |
| 0.0012        | 2375.0 | 19000 | 0.2389          | 94.5887       | 92.8571  |
| 0.0004        | 2500.0 | 20000 | 0.7480          | 91.7898       | 89.2857  |
| 0.0008        | 2625.0 | 21000 | 0.5444          | 91.7898       | 89.2857  |
| 0.0004        | 2750.0 | 22000 | 0.5347          | 91.7898       | 89.2857  |
| 0.0002        | 2875.0 | 23000 | 0.4793          | 91.7898       | 89.2857  |
| 0.0003        | 3000.0 | 24000 | 0.6247          | 91.7898       | 89.2857  |
| 0.0003        | 3125.0 | 25000 | 0.6901          | 91.7898       | 89.2857  |
| 0.0003        | 3250.0 | 26000 | 0.4848          | 91.7898       | 89.2857  |
| 0.0001        | 3375.0 | 27000 | 0.6783          | 91.7898       | 89.2857  |
| 0.0003        | 3500.0 | 28000 | 0.4926          | 91.7898       | 89.2857  |
| 0.0004        | 3625.0 | 29000 | 0.5892          | 91.7898       | 89.2857  |
| 0.0004        | 3750.0 | 30000 | 0.6808          | 91.7898       | 89.2857  |


### Framework versions

- Transformers 4.26.1
- Pytorch 2.0.0
- Datasets 2.7.1
- Tokenizers 0.13.2
