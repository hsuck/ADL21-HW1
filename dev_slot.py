import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader

from dataset import SeqClsDataset, SeqTaggingClsDataset
from model import SeqClassifier, SeqTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag2idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag2idx_path.read_text())

    data = json.loads(args.dev_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dev_dataloader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        collate_fn = dataset.collate_fn,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    num_class = dataset.num_classes
    model = SeqTagger(
        hidden_size = args.hidden_size,
        embeddings = embeddings,
        num_layers = args.num_layers,
        dropout = args.dropout,
        bidirectional = args.bidirectional,
        num_class = num_class
    )

    device = args.device
    model = model.to(device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict( ckpt )

    # TODO: predict dataset
    preds, groundtruth = [], []
    print('Start to evaluate...')
    print('')
    for i, datas in enumerate( dev_dataloader  ):
        inputs = datas['tokens']
        labels = datas['tags']
        groundtruth += labels

        inputs = torch.LongTensor( inputs ).to( device )

        # forward
        outputs = model( inputs )

        # calculate accuracy
        _, predictions = torch.max( outputs, dim = 2  )
        predictions = [ dataset.idxs2tag( prediction )[: datas['len'][i] ] for i, prediction in enumerate( predictions ) ]
        preds += predictions

    joint_acc, token_acc, total_len_tokens = 0, 0, 0
    for pred, label in list( zip( preds, groundtruth ) ):
        total_len_tokens += len( pred )
        corrects = sum([ int( p == l ) for p, l in zip( pred, label ) ])
        joint_acc += int( len( pred ) == corrects )
        token_acc += corrects

    print( "Joint Acc: {:3.6f}, ({}/{})".format( joint_acc / len( preds ), joint_acc, len( preds ) ) )
    print( "Token Acc: {:3.6f}, ({}/{})".format( token_acc / total_len_tokens, token_acc, total_len_tokens ) )
    print('')
    r = classification_report( y_true = groundtruth, y_pred = preds, scheme = IOB2, mode = 'strict' )
    print( r )

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dev_file",
        type=Path,
        help="Path to the eval file.",
        default = "./data/slot/eval.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default = './ckpt/slot/best_model.pt'
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
