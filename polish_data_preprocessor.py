import os
import re
from collections import Counter
from collections import defaultdict
import numpy as np

from tqdm import tqdm
import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.pl.examples import sentences




class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            self.nlp = spacy.load("pl_core_news_sm")

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words(
                '/home/arabica/Documents/pythonProject/GCN_PyTorch/PyTorch_TextGCN/polish.stopwords.txt'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result


def remove_less_word(lines_str, word_st):
    return " ".join([word for word in lines_str.split() if word in word_st])


class CorpusProcess:
    def __init__(self, train_dataset, test_dataset, dataset_name,encoding=None):
        corpus_path = "data/text_dataset/corpus"
        clean_corpus_path = "data/text_dataset/clean_corpus"
        if not os.path.exists(clean_corpus_path):
            os.makedirs(clean_corpus_path)

        self.dataset = dataset_name
        self.corpus_name = f"{corpus_path}/{dataset_name}.tsv"
        self.test_corpus_name = f"{corpus_path}/{test_dataset}.tsv"
        self.train_corpus_name = f"{corpus_path}/{train_dataset}.tsv"
        self.save_name = f"{clean_corpus_path}/{dataset_name}.txt"
        self.save_class_name = f"data/text_dataset/{dataset_name}.txt"
        self.context_dct = defaultdict(dict)

        self.encoding = encoding
        self.concat_files()
        self.clean_text()
        self.save_classes()

    def concat_files(self):
        train = pd.read_csv(self.train_corpus_name, sep="\t")[0:100]
        test = pd.read_csv(self.test_corpus_name, sep="\t")[0:100]
        train['split'] = 'train'
        test['split'] = 'test'
        df = pd.concat([train, test])
        df.to_csv(self.corpus_name, index=False, sep='\t')

    def save_classes(self):
        df = pd.read_csv(self.corpus_name, sep="\t")
        tmp_df = df.loc[:, ["split", "rating"]]
        tmp_df.to_csv(self.save_class_name, sep="\t", header=False)

    def clean_text(self):
        sp = StringProcess()
        word_lst = list()
        df = pd.read_csv(self.corpus_name, sep="\t")

        for _, item in df.iterrows():
            data = item['text'].strip()#.decode('latin1')
            data = sp.clean_str(data)
            if self.dataset not in {"mr"}:
                data = sp.remove_stopword(data)
            word_lst.extend(data.split())

        word_st = set()
        if self.dataset not in {"mr"}:
            for word, value in Counter(word_lst).items():
                if value < 5:
                    continue
                word_st.add(word)
        else:
            word_st = set(word_lst)

        doc_len_lst = list()
        with open(self.save_name, mode='w') as fout:
            for _, item in df.iterrows():
                lines_str = item['text'].strip()#.decode('latin1')
                lines_str = sp.clean_str(lines_str)
                if self.dataset not in {"mr"}:
                    lines_str = sp.remove_stopword(lines_str)
                    lines_str = remove_less_word(lines_str, word_st)

                fout.write(lines_str)
                fout.write(" \n")

                doc_len_lst.append(len(lines_str.split()))

        print("Average length:", np.mean(doc_len_lst))
        print("doc count:", len(doc_len_lst))
        print("Total number of words:", len(word_st))


def main():
    #CorpusProcess("R52")
    # CorpusProcess("20ng")
    # CorpusProcess("mr")
    # CorpusProcess("ohsumed")
    # CorpusProcess("R8")
    # pass
    CorpusProcess("train", "dev", "allegro")


if __name__ == '__main__':
    main()
