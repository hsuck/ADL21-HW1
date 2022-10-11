import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        collate_fn = dataset.collate_fn,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    num_class = dataset.num_classes
    model = SeqClassifier(
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
    result = []
    print('Start to test...')
    for i, datas in enumerate( test_dataloader  ):
        inputs = datas['text']
        inputs = torch.LongTensor( inputs  ).to( device  )

        # forward
        outputs = model( inputs  )

        # calculate accuracy
        _, predictions = torch.max( outputs, dim = 1  )
        predictions = [ dataset.idx2label( int( prediction ) ) for prediction in predictions ]
        ids = datas['id']
        result += list( zip( ids, predictions ) )

    # TODO: write prediction to file (args.pred_file)
    print( f'Writing predicttions to {args.pred_file}...' )
    with open( args.pred_file, 'w' ) as f:
        f.write('id,intent\n')
        for _id, prediction in result:
            f.write(f'{_id},{prediction}\n')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default = "./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default = './ckpt/intent/best_model.pt'
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
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
