---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- superglue-wic
metrics:
- accuracy
model-index:
- name: superglue-wic_lr3e-1_loralr1e-3_pl60_r30_st20000_ml256
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# superglue-wic_lr3e-1_loralr1e-3_pl60_r30_st20000_ml256

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on the superglue-wic dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3622
- Accuracy: 69.2790

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

| Training Loss | Epoch  | Step  | Validation Loss | Accuracy |
|:-------------:|:------:|:-----:|:---------------:|:--------:|
| 0.3173        | 5.88   | 1000  | 0.2830          | 50.7837  |
| 0.2996        | 11.76  | 2000  | 0.2803          | 54.5455  |
| 0.2892        | 17.65  | 3000  | 0.2833          | 53.2915  |
| 0.2629        | 23.53  | 4000  | 0.2818          | 61.1285  |
| 0.2411        | 29.41  | 5000  | 0.2754          | 64.5768  |
| 0.2326        | 35.29  | 6000  | 0.2806          | 66.1442  |
| 0.2244        | 41.18  | 7000  | 0.2930          | 65.8307  |
| 0.2174        | 47.06  | 8000  | 0.3419          | 63.6364  |
| 0.2114        | 52.94  | 9000  | 0.3116          | 67.3981  |
| 0.2082        | 58.82  | 10000 | 0.3191          | 67.3981  |
| 0.2032        | 64.71  | 11000 | 0.3268          | 67.3981  |
| 0.2004        | 70.59  | 12000 | 0.3389          | 67.7116  |
| 0.1969        | 76.47  | 13000 | 0.3505          | 66.1442  |
| 0.1955        | 82.35  | 14000 | 0.3456          | 68.9655  |
| 0.1926        | 88.24  | 15000 | 0.3436          | 68.3386  |
| 0.1908        | 94.12  | 16000 | 0.3505          | 68.9655  |
| 0.1867        | 100.0  | 17000 | 0.3615          | 68.9655  |
| 0.1842        | 105.88 | 18000 | 0.3622          | 69.2790  |
| 0.1851        | 111.76 | 19000 | 0.3583          | 69.2790  |
| 0.1831        | 117.65 | 20000 | 0.3617          | 68.9655  |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.13.1
- Datasets 2.10.1
- Tokenizers 0.13.2
