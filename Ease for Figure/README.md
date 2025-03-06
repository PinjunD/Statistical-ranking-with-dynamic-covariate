# [Statistical ranking with dynamic covariates](https://arxiv.org/abs/2406.16507)<br>
This repository allows you to reproduce all the figures from our paper.

## Installation
Clone the repository:

```bash
git clone https://github.com/PinjunD/Statistical-ranking-with-dynamic-covariate.git
cd Statistical-ranking-with-dynamic-covariate
cd Ease for Figure
```

Install the required Python packages:
```bash
-pip install numpy
-pip install pandas
-pip install matplotlib
-pip install joblib
```

## Structure
- **Figure/Table X.py**: Execute these scripts to generate the corresponding figures (e.g., **Figure/Table X.png**).
- **algorithm.py**: Contains the alternating maximization algorithm.
- **generator.py**: Generates multiple comparison hypergraphs.
- **basicfun.py**: Includes basic functions to streamline the process.
- **birthday.txt**: Lists the birthdays of all players discussed in the tennis data.
- **HSBM(300).txt/NURHM(300).txt**: Contains simulation results with 300 repeats for computational efficiency.
- **runs(preprocessing).csv/tennis(preprocessed).csv**: Preprocessed real data.
## Instruction
- Execute **Figure/Table X.py** scripts to generate the corresponding figures (e.g., **Figure/Table X.png**).
## Acknowledgements

The tennis data is sourced from [Jeff Sackmann](https://github.com/JeffSackmann/tennis-atp), while the horse racing data is from [Hong Kong horse-racing dataset](https://www.kaggle.com/datasets/gdaley/hkracing).

---
If you find this work useful, please consider citing it:



```bash
 @article{dong2024statistical,
  title={Statistical ranking with dynamic covariates},
  author={Dong, Pinjun and Han, Ruijian and Jiang, Binyan and Xu, Yiming},
  journal={arXiv preprint arXiv:2406.16507},
  year={2024}
}
```
