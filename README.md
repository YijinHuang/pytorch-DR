## pytorch-DR

- Implementation of team o_O solution for the Kaggle Diabetic Retinopathy Detection Challenge in pytorch.
- [Solution summary](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15617#latest-373487)

### Branches

- Branch "master" tries to apply new techniques to improve team o_O solution.
- Branch "reimplement" is the reimplementation of team o_O solution.
- Branch "two_cates" is used to do binary classification (Normal/DR).

### How to use

#### Data directory

Your should organize your dataset as following:

```
├── your_data_dir
	├── train
		├── 1
			├── a.jpg
			├── b.jpg
			├── ...
		├── 2
		├── 3
		├── 4
		├── 5
	├── val
	├── test
```

Here, `val` and `test` directory have the same structure of  `train`.  directories`1` - `5` means the severities of disease. 

#### Run

Most of hyperparameters and configures are in  `config.py`. You should choose `SMALL_NET_CONFIG`, `MEDIUM_NET_CONFIG` or `LARGE_NET_CONFIG` as `STEM_CONFIG` in main function of `main.py`. Function `stem` will train one inference network and function `blend` will train a ensemble network which is optional.

```python
def main():
    # network config, you should .
    STEM_CONFIG = SMALL_NET_CONFIG
    stem(STEM_CONFIG)
    
    # blend step config
    # BLEND_CONFIG = BLEND_NET_CONFIG
    # blend(BLEND_CONFIG, STEM_CONFIG)
```

Moreover, if you want to get the final large network, you should train small and medium network first. More detailed information are in `[o_O_solution_report.pdf](https://github.com/YijinHuang/pytorch-DR/blob/reimplement/o_O_solution_report.pdf)`.

### Result

This project is still in the process. The single large network can achieves 79.84% in EyePACs test set which is close to 80% that the author claims. You can use full ensemble methods that the author designed to get a better result but I haven't implement it yet.