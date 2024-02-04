---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- superglue-wsc.fixed
metrics:
- accuracy
model-index:
- name: superglue-wsc.fixed_lr3e-1_loralr1e-4_pl60_r30_st20000
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# superglue-wsc.fixed_lr3e-1_loralr1e-4_pl60_r30_st20000

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on the superglue-wsc.fixed dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2625
- Accuracy: 59.6154

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
- training_steps: 20000

### Training results

| Training Loss | Epoch   | Step  | Validation Loss | Accuracy |
|:-------------:|:-------:|:-----:|:---------------:|:--------:|
| 0.3121        | 55.56   | 1000  | 0.2739          | 40.3846  |
| 0.2952        | 111.11  | 2000  | 0.2687          | 50.0     |
| 0.2955        | 166.67  | 3000  | 0.2625          | 59.6154  |
| 0.2901        | 222.22  | 4000  | 0.2639          | 59.6154  |
| 0.2896        | 277.78  | 5000  | 0.2679          | 53.8462  |
| 0.2891        | 333.33  | 6000  | 0.2642          | 59.6154  |
| 0.2907        | 388.89  | 7000  | 0.2638          | 59.6154  |
| 0.2895        | 444.44  | 8000  | 0.2656          | 59.6154  |
| 0.2861        | 500.0   | 9000  | 0.2633          | 59.6154  |
| 0.285         | 555.56  | 10000 | 0.2684          | 44.2308  |
| 0.2852        | 611.11  | 11000 | 0.2637          | 59.6154  |
| 0.2853        | 666.67  | 12000 | 0.2646          | 59.6154  |
| 0.285         | 722.22  | 13000 | 0.2631          | 59.6154  |
| 0.2842        | 777.78  | 14000 | 0.2642          | 59.6154  |
| 0.285         | 833.33  | 15000 | 0.2633          | 59.6154  |
| 0.2837        | 888.89  | 16000 | 0.2671          | 53.8462  |
| 0.2838        | 944.44  | 17000 | 0.2642          | 59.6154  |
| 0.2837        | 1000.0  | 18000 | 0.2641          | 59.6154  |
| 0.2839        | 1055.56 | 19000 | 0.2638          | 59.6154  |
| 0.2833        | 1111.11 | 20000 | 0.2640          | 59.6154  |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.13.1
- Datasets 2.10.1
- Tokenizers 0.13.2
