# Political Bias Detection in News Articles

This project builds upon the research of:

> **Baly et al. (2020)**  
> *We Can Detect Your Bias: Predicting the Political Ideology of News Articles*  
> [https://doi.org/10.18653/v1/2020.acl-main.428](https://doi.org/10.18653/v1/2020.acl-main.428)



## Overview
The original research proposed models to predict the political bias of news articles using three ideology labels: Left, Center, and Right. In this project, I reimplement the baseline of Baly et al. (2020) and extend their work by adding:
- More recent transformer models (e.g., DistilBERT, ModernBERT)
- A statistical Logistic Regression baseline


The goal of this project is to reproduce and extend the results from the original paper, implementing various models for political bias detection in news articles. These additions aim to verify the challenges of generalization and robustness of models in political bias detection.

## Instructions

1. **Install dependencies**  
```bash pip install -r requirements.txt```.

2. **Run the scripts**  
Execute any of the scripts to train and evaluate the models.

3. **Select the data split**  
You can choose between the two available split strategies by changing the `split` flag inside the scripts to: `random` or `media`. 

## Results

The following tables compare the reproduced results to the original results reported by Baly et al. (2020).  
Values in parentheses (*) refer to the scores from the original paper.

### Results: Random Split

| Model             | Macro F1       | Accuracy      | MAE          |
|-------------------|----------------|---------------|--------------|
| BERT-base         | 0.81 (*80.19)  | 0.81 (*79.83) | 0.29 (*0.33) |
| DistilBERT        | 0.80           | 0.79          | 0.33         |
| ModernBERT        | **0.90**       | **0.87**      | **0.16**     |
| LSTM              | 0.63 (*65.50)  | 0.62 (*66.17) | 0.58 (*0.52) |
| Majority baseline | 0.31           | 0.32          | 0.92         |
| LogReg baseline   | 0.68           | 0.69          | 0.46         |

### Results: Media Split

| Model             | Macro F1       | Accuracy      | MAE          |
|-------------------|----------------|---------------|--------------|
| BERT-base         | 0.33 (*35.53)  | 0.38 (*36.75) | 0.83 (*0.90) |
| DistilBERT        | 0.40           | 0.44          | 0.74         |
| ModernBERT        | **0.48**       | **0.55**      | **0.55**     |
| LSTM              | 0.36 (*31.51)  | 0.39 (*32.30) | 0.83 (*0.97) |
| Majority baseline | 0.33           | 0.34          | 0.89         |
| LogReg baseline   | 0.38           | 0.42          | 0.75         |

## Citation

```bibtex
@inproceedings{baly2020we,
  author      = {Baly, Ramy and Da San Martino, Giovanni and Glass, James and Nakov, Preslav},
  title       = {We Can Detect Your Bias: Predicting the Political Ideology of News Articles},
  booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  series      = {EMNLP~'20},
  NOmonth     = {November},
  year        = {2020},
  pages       = {4982--4991},
  NOpublisher = {Association for Computational Linguistics}
}
```

## License
The dataset is released by the authoers under the **Apache License 2.0**  
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)