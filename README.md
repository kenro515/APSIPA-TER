# APSIPA-TER

This code is the implementation of Text Emotion Recognition (TER) with linguistic features.
The network model is BERT with a pretrained model parameter.

## How to use
0. Edit txt2tsv.py and preprocess your files 
```
python3 txt2tsv.py
```
1. Edit hyper_param.yaml
2. Run main.py
```
python3 main.py
```

## Reference

[1] GitHub_cl-tohoku/bert-japanese, https://github.com/cl-tohoku/bert-japanese (Last View: 2022-02-07)

[2] Huggingface_bert-japanese, https://huggingface.co/docs/transformers/model_doc/bert-japanese (Last View: 2022-02-07)

[3] Qiita-自然言語処理モデル（BERT）を利用した日本語の文章分類, https://qiita.com/takubb/items/fd972f0ac3dba909c293#bertforsequenceclassification (Last View: 2022-02-07)

## Paper

Ryotaro Nagase, Takahiro Fukumori and Yoichi Yamashita: ``Speech Emotion Recognition with Fusion of Acoustic- and Linguistic-Feature-Based Decisions, '' Proc. Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 725 -- 730, 2021.
