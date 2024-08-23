
import os
import numpy as np
import random
from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch_musa
import onnxruntime as ort
import onnxruntime.capi as ort_cap
import torch.nn.functional as F

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                Data[idx] = {
                    'sentence': sentence, 
                    'labels': labels
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collote_fn(batch_samples):
    batch_sentence, batch_labels  = [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_labels.append(sample['labels'])
    batch_inputs = tokenizer(
        batch_sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[s_idx][0] = -100
        batch_label[s_idx][len(encoding.tokens())-1:] = -100
        for char_start, char_end, _, tag in batch_labels[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, len(id2label))
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


batch_size = 64
input_len = 256
epoch_num = 1
item_num = 7

device = 'cpu'
print(f'Using {device} device')

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

categories = set()

test_data = PeopleDaily('../../data/china-people-daily-ner-corpus/example.test')

id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
config = AutoConfig.from_pretrained(checkpoint)
model = BertForNER.from_pretrained(checkpoint, config=config).to(device)

#model.load_state_dict(torch.load("epoch_3_valid_macrof1_95.812_microf1_95.904_weights.bin"))
bert_test = ort.InferenceSession("../../onnx_model/bert/bert_ner_fp16_64.onnx", providers=['MUSAExecutionProvider'])

def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            target_size = (batch_size, input_len)
            ori_num = X["input_ids"].size(0)
            padding = (0, target_size[1] - X["input_ids"].size(1), 0, target_size[0] - X["input_ids"].size(0))
            padded_ids = F.pad(X["input_ids"], padding, "constant", 0)
            padded_mask = F.pad(X['attention_mask'], padding, "constant", 0)
            padded_y = F.pad(y, padding, "constant", -100)
            padded_ids, padded_mask, y = padded_ids.to(device), padded_mask.to(device), padded_y.to(device)
            pred = bert_test.run(['output'], {'input_ids': np.array(padded_ids, dtype=np.int64), 'attention_mask': np.array(padded_mask, dtype=np.int64)})
            pred = torch.tensor(pred)
            pred = pred.view(-1, input_len, item_num)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()[:ori_num]

            labels = y.cpu().numpy().tolist()[:ori_num]
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in labels]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    return classification_report(
      true_labels, 
      true_predictions, 
      mode='strict', 
      scheme=IOB2, 
      output_dict=True
    )

for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    metrics = test_loop(test_dataloader, model)
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
print("Done!")