# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip instsall -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection training
```shell
python train_intent.py
```
## you can use following options to modify arguments of training
### for model
```shell
--hidden_size
--num_layers
--dropout
--bidirectional
```
### for optimizer
```shell
--lr
```
### for data loader
```shell
--batch_size
```
### use cpu or gpu
```shell
--device
```
### epoch
```shell
--num_epoch
```
## you can use following options to modify arguments of path
### directory to the dataset
```shell
--data_dir
```
