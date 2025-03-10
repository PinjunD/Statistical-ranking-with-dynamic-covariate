# Reproducibility for the numerical experiments
This repository contains the source code for all the numerical experiments in [Statistical ranking with dynamic covariates](https://arxiv.org/abs/2406.16507). For a better reproduction experience, we provide a detailed illustration as follows:

## 1. Installation
To get started, you can either download the source code directly or use Git to clone the repository.

```bash
git clone https://github.com/PinjunD/Statistical-ranking-with-dynamic-covariate.git
cd Statistical-ranking-with-dynamic-covariate
cd reproductions_codes
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
python "TableS2.py" # In HorceRacing
python "FigureS2.py" # In HorceRacing
python "FigureS3.py" # In HorceRacing
```

### 2.1. Demonstration:

<span style="font-size: 20px;">**Simulation**:<br></span> 
This pattern aims to confirm the uniform consistency of the maximum likelihood estimator (MLE) in the PlusDC model using simulated data, considering both the NURHM and HSBM random hypergraph models. (Please execute **FigureS1.py**)

* **FigureS1.py**: This executable file generates a hypergraph and edge-dependent covariates under specific setting, calculating the corresponding $l_\infty$ loss between estimator and true parameters. Under 300 repetitions, the file save the results in *FigureS1(a).png*, *FigureS1(b).png*, *FigureS1(c).png* and *FigureS1(d).png*.

<span style="font-size: 20px;">**Tennis**:<br></span>
This pattern seeks to investigate the aging effect on players in the sport of tennis to develop a dynamic "ability table" for the top 10 prominent players in history. After preprocessing the data, we select the optimal combination of Gaussian basis using the BIC criterion to approach the aging effect, comparing the results with the vanilla BT model (please execute **Figure5.py**). Additionally, we load information about players' birthdays to plot the "dynamic ability table" for the top 10 prominent players (please execute **Figure1.py**).

* **Figure5.py**: This executable file initiates 64 candidate combinations of Gaussian bases, calculating the coefficients of each basis using the PlusDC model. It then selects the optimal combination based on the BIC. The optimal bases and the aging effect function are plotted in *Figure5(a).png* and *Figure5(b).png*, respectively. Then, the file compares the players' rankings from the PlusDC model with those from the BT model, saving the results as *Figure(c).csv*.
* **Figure1.py**: This executable file utilizes the optimal bases (simlar to **Figure5.py**) to fit the PlusDC model, integrating the birthday information to plot the "dynamic ability table" in *Figure1.png.*

<span style="font-size: 20px;">**HorceRacing**:<br></span>
This pattern strives to study the impacts from actual weight, draw and public belief in horce racing. After traversing all the possible combinations of covariates, we compute the corresponding log-likelihood, AIC and BIC in *TableS2.csv* (please execute **TableS2.py**), comparing the results with vanilla PL model (please execute **FigureS2.py**). To futher examine the prediction performance, we conduct k-fold cross validation, and compare the results with the corresponding performance of the PL model and public belief (please execute **FigureS3.py**).

* **TableS2.py**: This executable file  computes the log-likelihood and AIC and BIC  corresponding to each combinition of covariates, saving the results in *TableS2.csv*.
* **FigureS2.py**: This executable file  compares the results from PlusDC model with those from PL model, saving the results in *FigureS2(right).csv* and *FigureS2(left).ong*.
* **FigureS3.py**: This executable file conducts k-fold cross validation, comparing the results with the corresponding performance of the PL model and public belief. The results have been plotted in *FigureS3(a).png*, *FigureS3(b).png* and *FigureS3(c).png*.

### 2.2.  Algorithm Core Functions
- **algorithm.py**: This file contains all the details in the alternating maximization algorithm.
- **generator.py**: This file provides a convenient way to generate hypergraph and edge-dependent covariates.
### 2.3. Dependencies
To reproduce the numerical results, we suggest install the the following dependencies in your Python environment
```bash
import numpy
import pandas
import matplotlib
import joblib

print(numpy.__version__)  # at least 1.26.4
print(pandas.__version__)  # at least 2.2.1
print(matplotlib.__version__)  # at least 3.9.0
print(joblib.__version__)  # at least 1.4.2
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
