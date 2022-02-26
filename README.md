# Near Instance Optimal Model Selection for Pure Exploration Linear Bandits

This repository contains the python code for our AISTATS 2022 paper **Near Instance Optimal Model Selection for Pure Exploration Linear Bandits**. Packages used include: numpy, sys, multiprocessing, pickle, time, logging and matplotlib.

Let `x = 0` (for experiment in Section 7) or `x = 1` (for experiment in Appendix F, we also set `max_iter = 500` in `algs_class.py` in this experiment). Use the following commands to reproduce our experiments.

```
python3 run_algs.py x
python3 plot.py x
```

On a cluster consists of two Intel® Xeon® Gold 6254 Processors, the runtime for experiment in Section 7 is around 20 minutes and the runtime for experiment in Appendix F is around 4.5 hours.

Part of the code is obtained/adapted from [GitHub - fiezt/Transductive-Linear-Bandit-Code](https://github.com/fiezt/Transductive-Linear-Bandit-Code). We thank authors of **Sequential Experimental Design for Transductive Linear Bandits** for their open-source code. We include a copy of their License in folder `LICENSE_OTHERS`.
