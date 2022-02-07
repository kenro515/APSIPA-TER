import re
import csv
import MeCab
import torch

from transformers import BertJapaneseTokenizer

m_c = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
m_t = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

trans_t = BertJapaneseTokenizer.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking')


def normalizing_JTES(p_text):
    p_text = re.sub(r'[0-9 ０-９]', '0', p_text)
    p_text = re.sub('\n', '', p_text)

    return p_text


def tokenizer_mecab(p_text):
    cleaner_list = []
    result = m_c.parseToNode(p_text)
    while result:
        if '名詞' in result.feature or \
            '形容詞' in result.feature or \
            '副詞' in result.feature or \
            '動詞' in result.feature and '助動詞' not in result.feature or \
                '記号' not in result.feature:
            cleaner_list.append(result.surface)
            result = result.next
        else:
            result = result.next

    return cleaner_list


def tokenizer_with_preprocessing(p_text):
    p_text = normalizing_JTES(p_text)
    ret = tokenizer_mecab(p_text)

    return ret


class dataloader_bert(object):
    def __init__(self, max_len=None, in_path=None):
        super(dataloader_bert, self).__init__()

        self.max_len = max_len
        self.in_path = in_path

    def get_train_data(self):
        input_ids = []
        attention_masks = []
        labels = []
        with open(self.in_path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                # If you want to normalize texts ...
                # raw_norm = tokenizer_with_preprocessing(row[0])

                encoded_dict = trans_t.encode_plus(
                    row[0],  # raw_norm
                    add_special_tokens=True,
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
                labels.append(int(row[1]))

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        labels = torch.tensor(labels)

        return input_ids, attention_masks, labels
