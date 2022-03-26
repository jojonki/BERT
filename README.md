# BERT
Pytorch implementation of BERT with [Sentencepiece tokenizer](https://github.com/google/sentencepiece).

I wrote the code as simple as possible so that you can understand BERT.
```
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
https://arxiv.org/abs/1810.04805
```

### Limitations
I have not been able to reproduce complete pretraining of BERT using this code because I do not have sufficient computational resources.
Therefore, there are no plans to release pre-trained models. If you have successfully pre-trained the model, please let me know. :)

## Data preparation for pre-training.
I prepared the data using the same procedure from https://github.com/yoheikikuta/bert-japanese.
The data must be line-by-line text format. In this case, I used Japanese texts, but any language will work since the tokenizer is Sentencepiece.


```bash
$ wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles-multistream.xml.bz2
$ bzip2 -d jawiki-latest-pages-articles-multistream.xml.bz2
$ python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles-multistream.xml
$ find text/ | grep wiki | awk '{system("cat "$0" >> wiki.txt")}'
$ sed -i -e '/^$/d; /<doc id/,+1d; s/<\/doc>//g' wiki.txt
$ sed -i -e 's/ *$//g; s/。\([^」|)|）|"]\)/。\n\1/g; s/^[ ]*//g' wiki.txt
$ sed -i -e '/^。/d' wiki.txt
$ sed -i -e 's/\(.*\)/\L\1/' wiki.txt
$ wc -l wiki.txt
 28536872 wiki.txt

$ mkdir corpora/jawiki/
$ mv wiki.txt corpora/jawiki/
```

## Pre-training
```bash
// from scratch
$ python3 run_pretraining.py  -c config/jawiki.yaml -e bert_pretraining

// resume (automatically find the latest checkpoint model)
$ python3 run_pretraining.py -c config/jawaiki.yaml -e bert_pretraining -r experiments/default/bert_pretraining

```

## Fine-tuning
I used [livedoor ニュースコーパス](https://www.rondhuit.com/download.html) and created `train.tsv/dev.tsv/test.tsv` according to https://github.com/yoheikikuta/bert-japanese/blob/master/notebook/finetune-to-livedoor-corpus.ipynb
You can use `BERTForSequenceClassification` class for classification tasks. 
```bash
$ python run_finetuning.py -c config/livedoor.yaml
```

## Try Masked Language Model
```
$ python run_mlm.py --checkpoint <path_to_checkpoint_model>
```

## References
- [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese)

