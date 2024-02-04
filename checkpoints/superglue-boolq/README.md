---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- superglue-boolq
metrics:
- accuracy
model-index:
- name: superglue-boolq_lr3e-1_loralr5e-3_pl40_r45_st30000
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# superglue-boolq_lr3e-1_loralr5e-3_pl40_r45_st30000

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on the superglue-boolq dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2088
- Accuracy: 80.2446

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

| Training Loss | Epoch  | Step  | Validation Loss | Accuracy |
|:-------------:|:------:|:-----:|:---------------:|:--------:|
| 0.3146        | 3.39   | 1000  | 0.2860          | 62.6300  |
| 0.3           | 6.78   | 2000  | 0.2783          | 62.6300  |
| 0.2968        | 10.17  | 3000  | 0.2789          | 62.6300  |
| 0.294         | 13.56  | 4000  | 0.2792          | 62.6300  |
| 0.2861        | 16.95  | 5000  | 0.2506          | 71.4985  |
| 0.2481        | 20.34  | 6000  | 0.2123          | 76.7584  |
| 0.2318        | 23.73  | 7000  | 0.2081          | 77.5535  |
| 0.224         | 27.12  | 8000  | 0.2081          | 78.3486  |
| 0.2122        | 30.51  | 9000  | 0.2046          | 78.2263  |
| 0.2081        | 33.9   | 10000 | 0.2045          | 78.6544  |
| 0.2015        | 37.29  | 11000 | 0.2051          | 79.6942  |
| 0.2006        | 40.68  | 12000 | 0.2015          | 78.7768  |
| 0.1977        | 44.07  | 13000 | 0.2040          | 79.0826  |
| 0.1946        | 47.46  | 14000 | 0.2052          | 79.3272  |
| 0.1911        | 50.85  | 15000 | 0.2042          | 80.0     |
| 0.1877        | 54.24  | 16000 | 0.2013          | 79.7554  |
| 0.1876        | 57.63  | 17000 | 0.2009          | 79.4495  |
| 0.1878        | 61.02  | 18000 | 0.1984          | 79.5719  |
| 0.184         | 64.41  | 19000 | 0.2045          | 79.8777  |
| 0.179         | 67.8   | 20000 | 0.2077          | 79.1437  |
| 0.1813        | 71.19  | 21000 | 0.2089          | 79.2049  |
| 0.1825        | 74.58  | 22000 | 0.2067          | 79.2661  |
| 0.1787        | 77.97  | 23000 | 0.2119          | 79.7554  |
| 0.1761        | 81.36  | 24000 | 0.2078          | 80.0612  |
| 0.1776        | 84.75  | 25000 | 0.2063          | 79.5719  |
| 0.1748        | 88.14  | 26000 | 0.2078          | 79.9388  |
| 0.1748        | 91.53  | 27000 | 0.2090          | 79.5719  |
| 0.1755        | 94.92  | 28000 | 0.2088          | 80.2446  |
| 0.1745        | 98.31  | 29000 | 0.2078          | 79.7554  |
| 0.1754        | 101.69 | 30000 | 0.2086          | 80.0     |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.13.1
- Datasets 2.10.1
- Tokenizers 0.13.2
