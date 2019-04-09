# Deep Multigrid

Implementation of the Deep Multigrid method, which optimizes restriction/prolongation operators and 
damp factor in Jacobi smoother in geometric multigrid method (GMG). 
The optimization is based on the reformulation of the geometric multigrid method in terms of the neural network with 
the specific architecture.
The objective function is the stochastic estimation of the spectral radius of the iteration matrix based on Gelfand's formula.
We provide demo notebooks for [Poisson](./poisson_test.ipynb) and [Helmholtz](./helmholtz_test.ipynb) equations, 
which show the performance of the proposed method compared to linear GMG and classical AMG 
from [PyAMG package](https://github.com/pyamg/pyamg).

## Run demo 

We use Python 3.5 in our experiments.
To run demo notebooks, you should install [autograd](https://github.com/HIPS/autograd) package for
automatic differentiation, standard numpy-scipy packages and matplotlib for simple visualization.
Also to speed-up computations we use [jit compilation with numba](https://numba.pydata.org/).

## Citing
If you use this code in your work, we kindly ask you to cite [the paper](https://arxiv.org/pdf/1711.03825.pdf)

```
@article{katrutsa2017deep,
  title={Deep Multigrid: learning prolongation and restriction matrices},
  author={Katrutsa, Alexandr and Daulbaev, Talgat and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1711.03825},
  year={2017}
}
```

Also presentation from the 19th Copper Mountain Conference On Multigrid Methods is available [here](https://easychair.org/smart-slide/slide/Ldwb#)
