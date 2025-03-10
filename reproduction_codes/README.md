# Reproducibility README for the numerical experiments in [Statistical ranking with dynamic covariates](https://arxiv.org/abs/2406.16507)
This repository contains the source code for the **alternating maximization algorithm** in , designed for statistical ranking with dynamic covariates. 

## 1. Installation
Please download the codes file, or install git bash and clone the repository as follows in Command:

```bash
git clone https://github.com/PinjunD/Statistical-ranking-with-dynamic-covariate.git
cd Statistical-ranking-with-dynamic-covariate
cd reproductions_codes
```

Install the required Python packages:
```bash
-pip install numpy
-pip install pandas
-pip install matplotlib
-pip install joblib
```

## 2. Repository Structure:
Below is a list of execution files for reproducing numerical results presented in Figure 1, Figure 8, in the main text, and Table 2, Figure 5, Figure 6, Figure 7 in the supplementary. For detailed procedures, please refer to Section 4 of the main text and Section S.3 of the supplementary material. Since we provide numerous simulations in Section S.3 of the supplementary material, we have included these four simulations in this repository to faciliate easier understanding and usage. These configurations are representative of the broader set of settings thereafter.
### 2.1. Demonstrations for Reproducing Numerical Results
- *Figure_5.py*: This execution file contains the codes for simulating parameters convergence. Specifically, we generate random hypergraph *NURHM* and *HSBM* with edge-depend covariates. With 300 repetitions, we compute the error of optimizer under $n\in\{200,400,600,800,1000\}$.
- *Table_2.py*: This execution file contains the codes for analyzing tennis data and plotting the Aging effect image.
### 2.2.  Algorithm Core Functions
- *algorithm.py*: This file contains all the details in the alternating maximization algorithm. 
- generator.py: 
### 3.1. Dependencies
To reproduce the numerical results, we suggest install the the following dependencies in your Python environment
### 3.2.  Reproducing Numerical Results
To run the simulations included in this repository, execute the following files:
```bash
python "Table_2.py"
python "Figure_1.py"
python "Figure_5.py"
python "Figure_6.py"
python "Figure_7.py"
python "Figure_8.py"
```

## 4. Acknowledgements

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
