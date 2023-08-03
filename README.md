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
    * `AWTC` [1] ([Ng et al. 2017](https://ieeexplore.ieee.org/abstract/document/7878527))
    * `HaLRTC` [2] ([Liu et al. 2012](https://ieeexplore.ieee.org/abstract/document/6138863))
    * `GTVM` [3] ([Chen et al. 2014](https://ieeexplore.ieee.org/document/6855213))
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

### References

[1] Ng, M.K.P., Yuan, Q., Yan, L. and Sun, J., 2017. An adaptive weighted tensor completion method for the recovery of remote sensing images with missing data. IEEE Transactions on Geoscience and Remote Sensing, 55(6), pp.3367-3381.

[2] Liu, J., Musialski, P., Wonka, P. and Ye, J., 2012. Tensor completion for estimating missing values in visual data. IEEE transactions on pattern analysis and machine intelligence, 35(1), pp.208-220.

[3] Chen, S., Sandryhaila, A., Lederman, G., Wang, Z., Moura, J.M., Rizzo, P., Bielak, J., Garrett, J.H. and Kovačevic, J., 2014, May. Signal inpainting on graphs via total variation minimization. In 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8267-8271). IEEE.
