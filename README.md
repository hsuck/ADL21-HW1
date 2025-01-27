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
--num_cnn
```
#### optimizer
```shell
--lr
--step_size
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
#### My settings
```shell
python train_intent.py --dropout 0.2 --num_layers 2 --hidden_size 512 --model GRU --lr 2e-4
```
others by default
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
#### My settings
```shell
python train_slot.py --dropout 0.3 --num_layers 2 --hidden_size 512 --model CNN_GRU --num_cnn 2 --lr 5e-4 --step_size 30
```
others by default
