# Reproducibility of the Numerical Experiments

This repository contains the source code for all the numerical experiments in [Statistical ranking with dynamic covariates](https://arxiv.org/abs/2406.16507). For a better reproduction experience, we provide a detailed illustration as follows:
```bash
Statistical-ranking-with-dynamic-covariate/
│
├── data/
│   ├── runs(preprocessed).csv
│   ├── tennis(preprocessed).csv
│   └── birthday.txt
│
├── package/
│   ├── __init__.py
│   ├── algorithm.py
│   └── generator.py
│
├── Simulation/
│   └── FigureS1.py
│
├── HorseRacing/
│   ├── TableS2.py
│   ├── FigureS2.py
│   └── FigureS3.py
│
├── Tennis/
│   ├── Figure1.py
│   └── Figure5.py
│
├── results/
│
├── Guideline.pdf
│
└── README.md
```
## 1. Installation
To get started, you can either download the source code directly or use Git to clone the repository.

```bash
git clone https://github.com/PinjunD/Statistical-ranking-with-dynamic-covariate.git
cd Statistical-ranking-with-dynamic-covariate
```

Ensure you have Python installed (preferably Python 3.x), then install the required packages using:

```bash
pip install numpy pandas matplotlib joblib
```

## 2. Repository:
This repository provides executable scripts for generating the figures and tables presented in our numerical experiments, including simulations, tennis data, and horse racing data. To reproduce our numerical results, simply execute the following code in the corresponding directory:

```bash

python "Figure1.py" # In Tennis
python "Figure5.py" # In Tennis
python "FigureS1.py" # In Simulation
python "TableS2.py" # In HorseRacing
python "FigureS2.py" # In HorseRacing
python "FigureS3.py" # In HorseRacing
```
Specifically, you can refer to the [*Guideline.pdf*](guideline.pdf).
### 2.1. Demonstration  
#### **Synthetic Data**  
This experiment verifies the uniform consistency of the maximum likelihood estimator (MLE) in the PlusDC model using simulated data, considering both the NURHM and HSBM random hypergraph models.  

📌 **To run:** Execute **FigureS1.py**  

- **FigureS1.py**:  
  - Generates a hypergraph with edge-dependent covariates under a specific setting.  
  - Computes the $\ell_\infty$-loss between the estimator and true parameters.  
  - Runs 300 repetitions and saves the results as:  
    - *FigureS1(a).pdf*  
    - *FigureS1(b).pdf*  
    - *FigureS1(c).pdf*  
    - *FigureS1(d).pdf*  

---

#### **Tennis**  
This experiment examines the age effect on tennis players and constructs a dynamic "ability table" for the top 10 players in history. After preprocessing the data, we select the optimal combination of Gaussian radial basis functions (RBFs) with different location and scale parameters using the BIC criterion to model the age effect and compare the results with the standard BT model.  

📌 **To run:**  
- Execute **Figure5.py** to determine the optimal Gaussian RBFs.  
- Execute **Figure1.py** to generate the "dynamic ability table".   

- **Figure5.py**:  
  - Tests 64 candidate models using subsets of all Gaussian RBFs.  
  - Computes basis coefficients using the PlusDC model.  
  - Selects the optimal combination using the BIC criterion.  
  - Plots and saves:  
    - *Figure5(a).pdf* (Optimal bases)  
    - *Figure5(b).pdf* (Estimated age effect)  
  - Compares player rankings obtained from the PlusDC and BT models, saving results in *Figure5(c).csv*.  

- **Figure1.py**:  
  - Uses the selected basis functions (from **Figure5.py**) to fit the PlusDC model.  
  - Integrates player birthdates to generate the "dynamic ability table".  
  - Saves the results as *Figure1.pdf*.  

---

#### **Horse Racing**  
This experiment investigates the incorporation of actual weight, draw position, and public belief in horse racing predictions using the PlusDC model. We evaluate different covariate combinations by computing log-likelihood, AIC, and BIC, and compare the results against the standard PL model. Additionally, we conduct k-fold cross-validation to assess predictive performance. 

📌 **To run:**  
- Execute **TableS2.py** to compute model selection criteria.  
- Execute **FigureS2.py** to compare the PlusDC model with the PL model.  
- Execute **FigureS3.py** for k-fold cross-validation and performance comparison.  

- **TableS2.py**:  
  - Computes log-likelihood, AIC, and BIC for all covariate combinations.  
  - Saves results in *TableS2.csv*.  

- **FigureS2.py**:  
  - Compares results from the PlusDC and PL models.  
  - Saves outputs as:  
    - *FigureS2(right).csv*  
    - *FigureS2(left).pdf*  

- **FigureS3.py**:  
  - Conducts k-fold cross-validation.  
  - Compares predictive performance with the PL model and public belief.  
  - Saves plots as:  
    - *FigureS3(a).pdf*  
    - *FigureS3(b).pdf*  
    - *FigureS3(c).pdf*  

### 2.2. Algorithm Core Functions  
- **algorithm.py**: Implements the alternating maximization algorithm.  
- **generator.py**: Provides functions for generating hypergraphs and edge-dependent covariates.  

### 2.3. Dependencies  
To reproduce the numerical results, ensure the following dependencies are installed in your Python environment: 

```bash
import numpy
import pandas
import matplotlib
import joblib

print(numpy.__version__)  # Required: at least 1.26.4
print(pandas.__version__)  # Required: at least 2.2.1
print(matplotlib.__version__)  # Required: at least 3.9.0
print(joblib.__version__)  # Required: at least 1.4.2
```


## 4. Acknowledgements

We gratefully acknowledge the sources of our datasets:  
- **Tennis data**: Provided by [Jeff Sackmann](https://github.com/JeffSackmann/tennis-atp).  
- **Horse racing data**: Sourced from the [Hong Kong horse racing dataset](https://www.kaggle.com/datasets/gdaley/hkracing).  

---

If you find this work useful and would like to cite it, please use the following reference: 



```bash
 @article{dong2024statistical,
  title={Statistical ranking with dynamic covariates},
  author={Dong, Pinjun and Han, Ruijian and Jiang, Binyan and Xu, Yiming},
  journal={arXiv preprint arXiv:2406.16507},
  year={2024}
}
```
