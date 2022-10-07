# Homework 1 ADL NTU

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

## Intent detection
```shell
python train_intent.py
```
### Options to modify arguments of training
#### model
```shell
--hidden_size
--num_layers
--dropout
--bidirectional
```
#### optimizer
```shell
--lr
```
#### data loader
```shell
--batch_size
```
#### cpu or gpu
```shell
--device
```
#### epoch
```shell
--num_epoch
```
### Options to modify arguments of path
#### directory to the dataset
```shell
--data_dir
```
#### directory to the dataset
```shell
--cache_dir
```
#### directory to save the model file
```shell
--ckpt_dir
```
## Slot tagging
```shell
python train_slot.py
```
slot tagging has the same options as intent detection, so you can follow above description to specify your training model's arguments 
