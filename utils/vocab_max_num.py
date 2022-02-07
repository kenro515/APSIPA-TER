import glob
from transformers import BertJapaneseTokenizer

trans_t = BertJapaneseTokenizer.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking')

if __name__ == '__main__':
    in_dir = "***"

    paths = sorted(glob.glob(in_dir))

    max_len = []
    for path in paths:
        f = open(path, 'r')
        text = f.readline()

        token_words = trans_t.tokenize(text)
        max_len.append(len(token_words))
        f.close()

    print('max_len:{}'.format(max(max_len)))