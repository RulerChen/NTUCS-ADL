# Applied Deep Learning

## Introduction

NTU CSIE 5431 ADL Programming Assignment 2024 Fall

## Homework

- [HW1](./hw1/README.md) : Chinese Extractive Question Answering (QA)
- [HW2](./hw2/README.md) : Chinese News Summarization (Title Generation)
- [HW3](./hw3/README.md) : Instruction Tuning (Classical Chinese)
- [Final](./final/README.md) : Can LLM learn with incoming streams of questions?

## Grade

- HW1 :

Private Baseline : 0.75846  

| Baseline | EM Score |
|:--------:|:--------:|
|  Public  |  0.787   |
| Private  | 0.79528  |

- HW2 :

Public Baseline : 22.5 / 8.5 / 20.5  
Private Baseline : 23.6 / 9.3 / 21.3  

| Baseline | Rouge1 | Rouge2 | RougeL |
|:--------:|:------:|:------:|:------:|
|  Public  | 24.90  | 9.983  | 22.35  |
| Private  | 24.58  | 9.906  | 22.14  |

- HW3 :

Public Baseline : 17.5  
Private Baseline : 24  

| Baseline | QLoRA |
|:--------:|:-----:|
|  Public  | 15.47 |
| Private  | 21.37 |

- Final :

| Baseline | Medical Diagnosis | Text-to-SQL Generation |
|:--------:|:-----------------:|:----------------------:|
|  Public  |      0.70521      |        0.35071         |
| Private  |      0.70918      |        0.20407         |

(Text to SQL 的 Private 有時可以接近 0.23，依電腦而異)
