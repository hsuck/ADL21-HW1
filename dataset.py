from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        sample_keys = list( samples[0].keys() )
        output = {}
        for key in sample_keys:
            if key == 'text':
                ## split each text by whitespace
                texts = [ sample['text'].split() for sample in samples ]
                output[key] = self.vocab.encode_batch( texts )
            else:
                output[key] = [ sample[key] for sample in samples ]

        return output
        #raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        super().__init__( data, vocab, label_mapping, max_len )
        self._idx2tag = { idx: tag for tag, idx in self.label_mapping.items() }
        ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        sample_keys = samples[0].keys()
        output = {}
        for key in sample_keys:
            if key == 'tokens':
                tokens = [ sample['tokens'] for sample in samples ]
                output['len'] = [ len( token ) for token in tokens ]
                output[key] = self.vocab.encode_batch( tokens, self.max_len )
            else:
                output[key] = [ sample[key] for sample in samples ]

        return output
        #raise NotImplementedError

    def tags2idx( self, tags: List[ int ] ):
        return [ self.label_mapping[tag] for tag in tags ]

    def idxs2tag( self, idxs: List[ int ] ):
        return [ self._idx2tag[ int( idx ) ] for idx in idxs ]
