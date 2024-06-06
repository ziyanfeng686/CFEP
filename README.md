# CFEP
CFEP is a ConFormer-based tool for predicting HLA class I and II epitopes.


## Installation
To install the required packages for running CFEP, please use the following command:
```
conda create -n <env_name> python==3.9
conda activate <env_name>
pip install -r requirements.txt
```
## Usage

There are five executable Python scripts in the directory: 

* `CFEPI_predict.py` is used for making predictions of HLA-I epitopes by using CFEPI model.
* `CFEPI_train.py` allows you to train  CFEPI on new data.
* `CFEPII_predict.py` is used for making predictions of HLA-II(HLA-DR) epitopes by using CFEPII model.
* `CFEPII_train.py` allows you to train CFEPII on new data.
* `evl_matric.py` allows you to evaluate model performance on test data after runing `CFEPI_predict.py` or `CFEPI_predict.py`.


Predictions will be written to `outputI.csv` and `outputII.csv` in the data folder. 

#### Supported Alleles

CFEPI only supports HLA-I while CFEPII supports HLA-DR. 
We use a uniform HLA HLA naming schemes, for example,`HLA-A*02:01`, `A*02:01`, `HLAA0201`, and `A0201` are all mapped to `HLA-A02:01`.


"# CFEP"  
