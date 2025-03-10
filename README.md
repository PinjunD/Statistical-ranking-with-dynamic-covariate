# [Statistical ranking with dynamic covariates](https://arxiv.org/abs/2406.16507)<br>

## Installation 
Please download the codes file, or install git bash and clone the repository as follows in Command:

```bash
git clone https://github.com/PinjunD/Statistical-ranking-with-dynamic-covariate.git
cd Statistical-ranking-with-dynamic-covariate
```

After that, you should proceed to install the fundamental packages for Python:
```bash
-pip install numpy
-pip install pandas
-pip install matplotlib
-pip install joblib
-pip install scipy
-pip install scikit-learn
```

## Structure
- **data**: This folder contains simulation results, as well as tennis and horse racing data. 
- **image**: This folder includes the figures from the simulation and real data analysis.
- **package**: This folder showcases the algorithm and synthetic data generator introduced in our paper.
- **video**: This folder contains two visualized videos that showcase the dynamic top-10 rankings derived from tennis data.
- **Ease for figures**: This folder provides a convenient way to reproduce all the figures in our paper. 
- **other files:**
  - **illustration.ipynb**: This file introduces a specific example to demonstrate our methods.
  - **simulation.ipynb**: This file details the process of simulating synthetic data.
  - **tennis.ipynb**: This file outlines the process of analyzing tennis data.
  - **horse racing.ipynb**: This file explains the process of analyzing horse racing data.

## Instruction
- You can open the *illustration.ipynb* file to gain a comprehensive understanding of the *package*, which includes some basic function, generator and algorithm.
- The *simulation.ipynb* contains conclusive numerical experiments. Specifically, we study the convergence rate of parameters, optimization algorithm and a tiny goodness-of-fit test.
- The part of real data analysis have presented in *horse racing.ipynb* and *tennis.ipynb* files, which include all the processes described in our paper.
## Acknowledgements

It's worth noting that the tennis data is sourced from [Jeff Sackmann](https://github.com/JeffSackmann/tennis-atp), while the horse racing data is from [Hong Kong horse-racing dataset](https://www.kaggle.com/datasets/gdaley/hkracing).

---
If you find our work beneficial for your research, we kindly ask you to consider citing our paper as follows:



```bash
 @article{dong2024statistical,
  title={Statistical ranking with dynamic covariates},
  author={Dong, Pinjun and Han, Ruijian and Jiang, Binyan and Xu, Yiming},
  journal={arXiv preprint arXiv:2406.16507},
  year={2024}
}
```
