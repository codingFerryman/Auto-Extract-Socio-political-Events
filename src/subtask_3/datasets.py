import sys
import ast
from collections import defaultdict
from typing import Tuple, Dict, List, Union
from pathlib import Path
from tqdm.auto import tqdm
import json
from cleantext.clean import clean

from allennlp.data import DatasetReader
from transformers import BigBirdTokenizer, BigBirdConfig, BigBirdForTokenClassification, PreTrainedModel, \
    PreTrainedTokenizer, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
CURRENT_DIRECTORY = Path(__file__).parent.resolve()

sys.path.append(str(Path(CURRENT_DIRECTORY, '..').resolve()))

from subtask_4 import bio_read
from subtask_3 import coref_read


class PreprocessedSpanBERTDataset(Dataset):
    def __init__(self, tokenizer=None, preprocessed_data_path=None, max_length=138):
        if preprocessed_data_path is None:
            preprocessed_data_path = Path(CURRENT_DIRECTORY, 'preprocessed_data.json')
        if not preprocessed_data_path.is_file():
            bio_data = load_bio()
            coref_data = load_coref()
            data = process_coref_data(coref_data, bio_data)
            save_to_file(data, save_path=preprocessed_data_path)
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.examples = load_coref(preprocessed_data_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        item.update(self.tokenize(**item))
        item["doc_key"] = "nw"
        # TODO
        # item = self.bio2cluster(item)
        return item

    def tokenize(self, tokens, tags, **kwargs):
        sentences = []
        speakers = []
        tags_map = []
        subtoken_map = []
        for tokens_instance, tags_instance in zip(tokens, tags):
            sentence = ["[CLS]"]
            speaker = ["[SPL]"]
            tag_map = ["O"]
            id_map = []
            subtoken_id = 0
            for token, tag in zip(tokens_instance, tags_instance):
                subtokens = self.tokenizer.tokenize(token)
                sentence.extend(subtokens)
                speaker.extend(['-']*len(subtokens))
                tag_map.extend([tag]*len(subtokens))
                id_map.extend([subtoken_id]*len(subtokens))
                subtoken_id += 1
            sentence.append("[SEP]")
            speaker.append(["[SPL]"])
            tag_map.append("O")
            id_map.append(id_map[-1])
            sentences.append(sentence)
            speakers.append(speaker)
            tags_map.append(tag_map)
            subtoken_map.append(id_map)

        return {"sentences": sentences, "speakers": speakers, "tags": tags_map, "subtoken_map": subtoken_map}

    def bio2cluster(self, item: dict):
        sentence_no = item['sentence_no']
        cluster2id = {k: v for k, v in enumerate(sentence_no)}
        id2cluster = {v: k for k, v in enumerate(sentence_no)}
        subtoken_cluster_id = 0
        for cluster in item['event_clusters']:
            cluster = [cluster2id[c] for c in cluster]
            # TODO


def load_bio(data_path=None):
    if data_path is None:
        _bio_path = Path(CURRENT_DIRECTORY, '../..', 'data', 'subtask4-token', 'en-train.txt').resolve()
        assert _bio_path.is_file()
        data_path = _bio_path
    _bio_data = bio_read(data_path)
    _bio_data = [(sent[1:], tag[1:]) for sent, tag in _bio_data]
    result = preprocess_bio_data(_bio_data)
    return result


def load_coref(data_path=None):
    if data_path is None:
        _coref_path = Path(CURRENT_DIRECTORY, '../..', 'data', 'subtask3-coreference', 'en-train.json').resolve()
        assert _coref_path.is_file()
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

        _idx = 0
        _words = []
        _tags = []
        for _word, _tag in zip(document[0], document[1]):
            if _idx not in _delimiter_location:
                _words.append(_word)
                _tags.append(_tag)
            else:
                _key = "".join(_words)
                _key = preprocess_key_clean(_key)
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

    token_past = tokens[0][1:]
    tokens_result = [token_past]
    tags = tags[1:-1]
    tags_past = tags[0]
    tags_result = [tags_past]
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
    tokens_result = []
    tags_result = []
    for sent in sentences:
        _fetch_key = "".join(sent)
        _fetch_key = preprocess_key_clean(_fetch_key)
        _fetched = bio_data[_fetch_key]
        if len(_fetched) == 0:
            tokens, tags = _predict_failed_fetch(sent, **kwargs)
            tokens_result.append(tokens)
            tags_result.append(tags)
        else:
            tokens_result.append(_fetched['tokens'][0])
            tags_result.append(_fetched['tags'][0])
    _data = {'tokens': tokens_result, 'tags': tags_result}
    return _data


def get_model_tokenizer(bio_model_path: str = None):
    if bio_model_path is None:
        _model_path = Path(CURRENT_DIRECTORY, '../..', 'models', 'bigbird-subtask4').resolve()
        assert _model_path.is_dir()
    config = BigBirdConfig.from_pretrained(_model_path)
    model = BigBirdForTokenClassification.from_pretrained(_model_path, config=config).to(device)
    tokenizer = BigBirdTokenizer.from_pretrained(_model_path)
    return model, tokenizer


def process_coref_data(documents: List[Dict], bio_data: dict) -> List[Dict]:
    labels = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger",
              "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O",
              "PAD"]
    id2label = {i: j for i, j in enumerate(labels)}

    model, tokenizer = get_model_tokenizer()

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
        save_path = Path(CURRENT_DIRECTORY, 'preprocessed_data.json').resolve()

    with open(save_path, "w", encoding="utf-8") as f:
        for doc in processed_data:
            f.write(json.dumps(doc) + "\n")


def analyze_data(dataset_input):
    sent_count = 0
    target_count = 0
    trigger_count = 0
    participant_count = 0
    organizer_count = 0
    no_tag_count = 0
    max_length = 0
    for data in dataset_input:
        tags = data['tags']
        sent_count += len(tags)
        for tags_instance in tags:
            tags_instance = [ins.split('-')[1] for ins in tags_instance if '-' in ins]
            tags_instance_len = len(tags_instance)

            if tags_instance_len > max_length:
                max_length = tags_instance_len
            flag = False
            if 'trigger' in tags_instance:
                trigger_count += 1
                flag = True
            if 'participant' in tags_instance:
                participant_count += 1
                flag = True
            if 'organizer' in tags_instance:
                organizer_count += 1
                flag = True
            if 'target' in tags_instance:
                target_count += 1
                flag = True
            if flag is False:
                no_tag_count += 1
    print(f"Sentences' max length: {max_length}")
    print("The ratio of sentences having ...")
    print(f"trigger: \t {trigger_count/sent_count: 2.2%}")
    print(f"participant: \t {participant_count/sent_count: 2.2%}")
    print(f"organizer: \t {organizer_count/sent_count: 2.2%}")
    print(f"target: \t {target_count/sent_count: 2.2%}")
    print(f"Data without any tags above: \t {no_tag_count/sent_count: 2.2%}")

    """
    Sentences' max length: 154
    The ratio of sentences having ...
    trigger: 	  97.64%
    participant: 	  56.84%
    organizer: 	  22.78%
    target: 	  29.10%
    Data without any tags above: 	  1.20%
    """


if __name__ == "__main__":
    dataset = PreprocessedSpanBERTDataset()
    analyze_data(dataset)
    pass
