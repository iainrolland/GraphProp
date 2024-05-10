## GraphProp

### Requirements

> * `matplotlib==3.5.1`
> * `numpy==1.22.3`
> * `Pillow==9.0.1`
> * `Pillow==9.5.0`
> * `scikit_learn==1.0.2`
> * `scipy==1.8.0`
> * `tqdm==4.64.0`
> * `xarray==2023.2.0`

Install packages with pip:
```bash
pip install -r requirements.txt
```
## Demo
`main.py` can be used to demo GraphProp as well as the benchmarks. The following arguments are available:

* `--method` one of:
    * `GraphProp` (our method)
* `--pattern` one of:
    * `SLC-off`
    * `partial-ovelap`
* `--plot` one of:
    * `True` (save results then show them in `matplotlib` window)
    * `False` (save results then exit)

For example, to run GraphProp on the SLC-off pattern and plot the results, run:
    
```bash
python main.py --method GraphProp --pattern SLC-off --plot True
```

Within the directory named `experiments` the results will be saved. Saved outputs are as follows:

> * `{method}_demo.json`: stores runtime parameters
> * `{method}_demo_0.log`: stores logged outputs including MSE, RMSE, MAE and mPSNR
> * `{method}_demo_output.npy`: stores method's output as a 4th-order tensor
> * `{method}_demo_mask.npy`: stores the mask used for the experiment location of the entries which where hidden
