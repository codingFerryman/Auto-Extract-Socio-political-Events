import os
import sys
from collections import defaultdict
from typing import Tuple, Dict, List, Union
from pathlib import Path
from tqdm.auto import tqdm
import json
from cleantext.clean import clean

from allennlp.data import DatasetReader
from transformers import BigBirdTokenizer, BigBirdConfig, BigBirdForTokenClassification, PreTrainedModel, \
    PreTrainedTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from subtask_4 import bio_read
from subtask_3 import coref_read


class SpanBERTDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]


def load_bio(data_path=None):
    if data_path is None:
        _file_path = os.path.dirname(__file__)
        _bio_path = os.path.abspath(os.path.join(_file_path, '../..', 'data', 'subtask4-token', 'en-train.txt'))
        assert Path(_bio_path).is_file()
        data_path = _bio_path
    _bio_data = bio_read(data_path)
    _bio_data = [(sent[1:], tag[1:]) for sent, tag in _bio_data]
    result = preprocess_bio_data(_bio_data)
    return result


def load_coref(data_path=None):
    if data_path is None:
        _file_path = os.path.dirname(__file__)
        _coref_path = os.path.abspath(os.path.join(_file_path, '../..', 'data', 'subtask3-coreference', 'en-train.json'))
        assert Path(_coref_path).is_file()
        data_path = _coref_path
    result = coref_read(data_path)
    return result


def preprocess_key_clean(sent: str) -> str:
    return clean(
        sent,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=True,
        no_numbers=True,
        no_digits=True,
        no_punct=True,
        replace_with_number="",
        replace_with_digit="",
        replace_with_punct="",
    ).replace(" ", "")


def preprocess_bio_data(documents: List[Tuple]) -> Dict:
    result = defaultdict(lambda: defaultdict(list))
    for document in tqdm(documents, desc="Preprocessing BIO data in subtask4: "):
        _sent, _ = document
        _delimiter_location = [i for i, x in enumerate(_sent) if x == '[SEP]']
        _filtered_sent = list(filter(('[SEP]').__ne__, _sent))
        _key = "\n".join(_filtered_sent)
        _key = preprocess_key_clean(_key)

        _idx = 0
        _words = []
        _tags = []
        for _word, _tag in zip(document[0], document[1]):
            if _idx not in _delimiter_location:
                _words.append(_word)
                _tags.append(_tag)
            else:
                result[_key]["tokens"].append(_words)
                result[_key]["tags"].append(_tags)
                _words = []
                _tags = []
            _idx += 1
    return result


def _predict_failed_fetch(sent: str, id2label: dict, bio_model: PreTrainedModel, bio_tokenizer: PreTrainedTokenizer):
    tokens = bio_tokenizer.tokenize(sent)
    input_ids = bio_tokenizer.encode(sent, return_tensors="pt")
    logits = bio_model(input_ids).logits[0]
    preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().tolist()
    result = [id2label[i] for i in preds]
    tokens, result = _decode_subtoken_tags(tokens, result)
    return tokens, result


def _decode_subtoken_tags(tokens, tags):
    tokens_result = []
    tags_result = []
    token_past = tokens[0][1:]
    tags_past = tags[0]
    for token, tag in zip(tokens[1:], tags[1:]):
        if not token.startswith('▁'):
            token_past += token
            tags_past = tag
        else:
            tokens_result.append(token_past)
            tags_result.append(tags_past)
            token_past = token[1:]
            tags_past = tag
    if tokens[-1].startswith('▁'):
        tokens_result.append(tokens[-1][1:])
        tags_result.append(tags[-1])
    return tokens_result, tags_result


def fetch_bio(sentences: List[str], bio_data: dict, **kwargs):
    _fetch_key = "".join(sentences)
    _fetch_key = preprocess_key_clean(_fetch_key)
    _data = bio_data[_fetch_key]
    if len(_data) == 0:
        tokens_result = []
        tags_result = []
        for sent in sentences:
            tokens, tags = _predict_failed_fetch(sent, **kwargs)
            tokens_result.append(tokens)
            tags_result.append(tags)
        _data = {'tokens': tokens_result, 'tags': tags_result}
    return _data


def process_coref_data(documents: List[Dict], bio_data: dict, bio_model_path: str = None) -> List[Dict]:
    labels = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger",
              "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O",
              "PAD"]
    id2label = {i: j for i, j in enumerate(labels)}

    if bio_model_path is None:
        _file_path = os.path.dirname(__file__)
        _model_path = os.path.abspath(os.path.join(_file_path, '../..', 'models', 'bigbird-subtask4'))
        assert Path(_model_path).is_dir()
    config = BigBirdConfig.from_pretrained(_model_path)
    model = BigBirdForTokenClassification.from_pretrained(_model_path, config=config).to(device)
    tokenizer = BigBirdTokenizer.from_pretrained(_model_path)

    result = []
    for document in tqdm(documents, desc="Fetching BIO information: "):
        sentences = document["sentences"]
        _fetched = fetch_bio(sentences=sentences, id2label=id2label, bio_data=bio_data, bio_model=model,
                             bio_tokenizer=tokenizer)
        document.update(_fetched)
        result.append(document)
    return documents


def save_to_file(processed_data: List[Dict], save_path: Union[str, Path] = None):
    if save_path is None:
        _file_path = os.path.dirname(__file__)
        save_path = os.path.abspath(os.path.join(_file_path, 'preprocessed_data.json'))

    with open(save_path, "w", encoding="utf-8") as f:
        for doc in processed_data:
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    bio_data = load_bio()
    coref_data = load_coref()
    data = process_coref_data(coref_data, bio_data)
    save_to_file(data)
