# Code

This is the open source code of the paper "Machine Learning Predicts Conductivity in 2D Metal Chalcogenides for Designing Optimal Electromagnetic Wave Absorbing Materials".

## run

The experiments are conducted on Win10 with Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz.

You should run the following commands to get the results:

```bash
cd code
pip install -r requirements.txt
```

Run `train.py` to train the model and get the classification accuracy.
```bash
python train.py
```

Run `metrics.py` to calculate the metrics of different models.
```bash
python metrics.py
```

Run `feature.py` to calculate the importance of different features.
```bash
python feature.py
```

Run `pdp.py` to get the partial dependence plot of features.
```bash
python pdp.py
```

Run `conf_matrix.py` to get the confusion matrix of predictions.
```bash
python conf_matrix.py
```

Run `corr_matrix.py` to get the correlation matrix of features.
```bash
python corr_matrix.py
```
