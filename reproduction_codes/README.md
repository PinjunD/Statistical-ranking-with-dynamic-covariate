# Reproducibility for the numerical experiments in [Statistical ranking with dynamic covariates](https://arxiv.org/abs/2406.16507)
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

## 2. Repository:
We summarize a list of execution files for plotting figured presented in simulation, tennis data and horce-racing data. To reproduce our numerical results, you just need to execute the following code in the correponding folder:
```bash

python "Figure1.py" # In Tennis
python "Figure5.py" # In Tennis
python "FigureS1.py" # In Simulation
python "TableS2.py" # In HorceRacing
python "FigureS2.py" # In HorceRacing
python "FigureS3.py" # In HorceRacing
```

### 2.1. Demonstrations:

<span style="font-size: 20px;">**Simulation**:</span> 
This pattern aims to confirm the uniform consistency of the maximum likelihood estimator (MLE) in the PlusDC model using simulated data, considering both the NURHM and HSBM random hypergraph models. (Please execute **FigureS1.py**)

* **FigureS1.py**: This executable file generates a hypergraph and edge-dependent covariates under specific setting, calculating the corresponding $l_\infty$ loss between estimator and true parameters. Under 300 repetitions, the file save the results in *FigureS1(a).png*, *FigureS1(b).png*, *FigureS1(c).png* and *FigureS1(d).png*.

<span style="font-size: 20px;">**Tennis**:</span>
This pattern seeks to investigate the aging effect on players in the sport of tennis to develop a dynamic "ability table" for the top 10 prominent players in history. After preprocessing the data, we select the optimal combination of Gaussian basis using the BIC criterion to approach the aging effect, comparing the results with the vanilla BT model (please execute **Figure5.py**). Additionally, we load information about players' birthdays to plot the "dynamic ability table" for the top 10 prominent players (please execute **Figure1.py**).

* **Figure5.py**: This executable file initiates 64 candidate combinations of Gaussian bases, calculating the coefficients of each basis using the PlusDC model. It then selects the optimal combination based on the BIC. The optimal bases and the aging effect function are plotted in *Figure5(a).png* and *Figure5(b).png*, respectively. Then, the file compares the players' ranking from PlusDC with that from BT, and saves the results as *Figure(c).csv*.
* **Figure1.py**: This executable file utilizes the optimal bases to fit PlusDC model, combining the birthday information to plot the "dynamic ability table" *Figure1.png*.


<span style="font-size: 20px;">**HorceRacing**:</span>


* **Table_2.py**: 
* **Figure_6.py**: 
* **Figure_7.py**: 





This execution file contains the codes for analyzing tennis data and plotting the Aging effect image.


### 2.2.  Algorithm Core Functions
- **algorithm.py**: This file contains all the details in the alternating maximization algorithm. 
- **generator.py**: 
### 3.1. Dependencies
To reproduce the numerical results, we suggest install the the following dependencies in your Python environment
### 3.2.  Reproducing Numerical Results


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
