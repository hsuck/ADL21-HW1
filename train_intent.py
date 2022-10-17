import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import trange
import torch.optim as optim

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

import numpy as np
import random

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(
        dataset = datasets[TRAIN],
        batch_size = args.batch_size,
        collate_fn = datasets[TRAIN].collate_fn,
        shuffle = True
    )

    dev_dataloader = DataLoader(
        dataset = datasets[DEV],
        batch_size = args.batch_size,
        collate_fn = datasets[DEV].collate_fn,
        shuffle = True
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    num_class = datasets[TRAIN].num_classes
    model = SeqClassifier(
        model = args.model,
        hidden_size = args.hidden_size,
        embeddings = embeddings,
        num_layers = args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = num_class,
        num_cnn = args.num_cnn
    )

    model = model.to( device )

    criterion = nn.CrossEntropyLoss()

    # TODO: init optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr = args.lr,
        weight_decay = 1e-5
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = args.step_size,
        gamma = 0.1
    )

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_acc, train_losses, dev_acc, dev_losses = 0, 0, 0, 0
        for i, datas in enumerate( train_dataloader ):
            # get inputs&labels from train dataloader
            inputs = datas['text']
            labels = datas['intent']
            labels = [ intent2idx[label] for label in labels ]

            inputs = torch.LongTensor( inputs ).to( device )
            labels = torch.LongTensor( labels ).to( device )

            # forward
            outputs = model( inputs )

            # calculate loss
            loss = criterion( outputs, labels )

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            _, predictions = torch.max( outputs, dim = 1 )
            train_acc += ( predictions.cpu() == labels.cpu() ).sum().item()
            train_losses += loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for i, datas in enumerate( dev_dataloader ):
                inputs = datas['text']
                labels = datas['intent']
                labels = [ intent2idx[label] for label in labels ]

                inputs = torch.LongTensor( inputs ).to( device )
                labels = torch.LongTensor( labels ).to( device )

                # forward
                outputs = model( inputs )

                # calculate loss
                loss = criterion( outputs, labels )


                # calculate accuracy
                _, predictions = torch.max( outputs, dim = 1 )
                dev_acc += ( predictions.cpu() == labels.cpu() ).sum().item()
                dev_losses += loss.item()

            print( '\nTrain Acc: {:3.6f} Loss: {:3.6f} | Dev Acc: {:3.6f} loss: {:3.6f}'.format(
                train_acc / len( datasets[TRAIN] ),
                train_losses / len( train_dataloader ),
                dev_acc / len( datasets[DEV] ),
                dev_losses / len( dev_dataloader ) ) )

            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save( model.state_dict(), args.ckpt_dir / ( args.model + "_best_model.pt" ) )
                print('saving model...')

        scheduler.step()
        print( 'Best acc: {:3.6f}'.format( best_acc / len( datasets[DEV] ) ) )
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument( "--model", type = str, default = 'GRU' )
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument( "--num_cnn", type = int, default = 1 )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=10)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed = 1234
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed( seed )
    np.random.seed( seed )
    random.seed( seed )
    main(args)
