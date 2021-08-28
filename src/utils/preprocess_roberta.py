"""Preprocessing code: DOC-SENTENCES & FULL-SENTENCES"""
from typing import List

from datasets import concatenate_datasets, load_dataset
import datasets
from transformers import RobertaTokenizerFast

from tqdm.contrib import tenumerate
from tqdm import tqdm

from pathlib import Path

from sentencizer import Sentencizer

import numpy as np
import torch

from argparse import ArgumentParser
parser = ArgumentParser(description="Preprocess datasets for RoBERTa")
parser.add_argument("-p", "--path", help="(str) Where to save or where to load from?", 
                    default=None)
parser.add_argument("--disable_tqdm", help="(bool) If `True`, a progress bar will be disabled.",
                    default=False)
parser.add_argument("--mask_prob", help="(float) Ratio of masked tokens",
                    default=0.15)
args = parser.parse_args()


# NLTK's stopword list
# https://www.nltk.org/nltk_data/
stopword_list = {"i","me","my","myself","we","our","ours","ourselves","you","you're",
                "you've","you'll","you'd","your","yours","yourself","yourselves",
                "he","him","his","himself","she","she's","her","hers","herself",
                "it","it's","its","itself","they","them","their","theirs",
                "themselves","what","which","who","whom","this","that","that'll",
                "these","those","am","is","are","was","were","be","been","being",
                "have","has","had","having","do","does","did","doing","a","an",
                "the","and","but","if","or","because","as","until","while","of",
                "at","by","for","with","about","against","between","into",
                "through","during","before","after","above","below","to","from",
                "up","down","in","out","on","off","over","under","again","further",
                "then","once","here","there","when","where","why","how","all",
                "any","both","each","few","more","most","other","some","such",
                "no","nor","not","only","own","same","so","than","too","very","s",
                "t","can","will","just","don","don't","should","should've","now",
                "d","ll","m","o","re","ve","y","ain","aren","aren't","couldn",
                "couldn't","didn","didn't","doesn","doesn't","hadn","hadn't",
                "hasn","hasn't","haven","haven't","isn","isn't","ma","mightn",
                "mightn't","mustn","mustn't","needn","needn't","shan","shan't",
                "shouldn","shouldn't","wasn","wasn't","weren","weren't","won",
                "won't","wouldn","wouldn't"}

stopword_id_list = {5632, 5634, 5, 519, 8, 9, 10, 11, 524, 1529, 14, 34318, 16, 
                    13, 18, 531, 5652, 6677, 16918, 23, 24, 19, 21, 25, 28, 29, 
                    30, 31, 32, 33, 34, 548, 37, 7, 39, 40, 35369, 41, 1065, 42, 
                    45, 47, 560, 49, 50, 51, 52, 53, 54, 13367, 56, 1075, 58, 59, 
                    1594, 61, 62, 63, 55, 64, 66, 579, 26692, 69, 3654, 2629, 70, 
                    581, 1610, 75, 15, 77, 2126, 79, 71, 81, 7761, 37460, 84, 
                    7254, 87, 88, 2137, 89, 596, 1116, 605, 90, 95, 608, 97, 98, 
                    99, 2661, 102, 103, 617, 106, 10859, 21098, 109, 110, 114, 
                    627, 37492, 117, 17526, 119, 27768, 630, 118, 123, 8310, 
                    13437, 122, 127, 35457, 129, 136, 2185, 137, 139, 141, 142, 
                    143, 144, 145, 658, 147, 148, 21651, 150, 8338, 149, 7325, 
                    1694, 159, 160, 23201, 162, 25252, 7333, 167, 1193, 1705, 
                    683, 172, 2220, 8877, 4783, 12465, 179, 35507, 182, 25784, 
                    12473, 700, 31934, 4288, 24770, 7877, 197, 25800, 3785, 4297, 
                    29902, 209, 31954, 1235, 5844, 214, 215, 11990, 218, 41690, 
                    219, 222, 223, 4321, 1250, 227, 9442, 2282, 13040, 16625, 
                    240, 241, 15605, 17143, 22268, 7424, 769, 258, 259, 260, 
                    9475, 1794, 261, 20235, 26895, 7443, 30484, 276, 281, 282, 
                    10010, 3355, 8987, 57, 18212, 1322, 4395, 39212, 9006, 37167, 
                    2864, 1843, 308, 15157, 15159, 6968, 2362, 1851, 30012, 14656, 
                    29510, 23367, 326, 9226, 348, 349, 11613, 351, 350, 354, 36195, 
                    874, 367, 5488, 368, 3955, 20343, 8569, 890, 17276, 1916, 385, 
                    1409, 6025, 8585, 399, 1423, 23444, 405, 37782, 5525, 5016, 
                    17304, 12186, 14746, 11672, 3486, 415, 417, 12196, 5030, 424, 
                    938, 9131, 939, 429, 10669, 12724, 25017, 19385, 1979, 17341, 
                    24001, 965, 454, 1990, 456, 1481, 463, 2512, 10705, 8663, 
                    1495, 473, 9178, 475, 8155, 2527, 5087, 44513, 995, 3559, 
                    2025, 14314, 1003, 1017, 23033, 506, 1021}

"""
# Corresponding RoBERTa token list
['with', 'ours', 'this', 'doesn', "'s", 'than', 'will', 'such', 's', 've', 'them', 
'to', 'om', 'self', 'if', 'does', 'its', "'ll", 'not', 'be', "'t", 'our', 'from', 
'during', 'don', 'over', 't', 'of', 'w', 'a', 'irs', 'here', 'the', 'few', 'now', 
'm', 'those', 'won', 'i', 'because', 'might', 'o', 'up', 'only', 'nor', 'was', 
'we', 'again', 'once', 'has', 'sh', 'she', 'before', 'my', 'some', 'been', 'in', 
'between', 'into', 'he', 'didn', 'him', 'most', 'why', 'but', 'any', 'these', 
'have', "'re", 'same', 'y', 'more', 'being', 'all', 'then', 'your', 're', 'too', 
'both', 'against', 'can', 'me', 'aren', 'an', 'on', 'while', 'must', 'other', 
'need', 'as', 'n', 'they', 'just', 'own', 'are', 'urther', 'about', 'having', 
'her', 'haven', 'you', 'no', 'ain', 'doing', 'would', 'until', "'ve", 'wh', 'is', 
'below', 'or', 'which', 'down', 'where', 'll', 'ma', 'should', 'by', 'that', 
'there', 'above', 'very', 'it', 'each', 'could', 'through', 'had', 'when', 'do', 
'at', 'd', 'what', 'am', 'after', 'his', 'their', 'were', 'did', 'for', 'and', 
'hers', 'eren', 'how', 'who', 'so', 'under', 'selves', 'out', 'off', 'f', "'d"]

# Note:
# A RoBERTa tokeniser (byte-level Byte-Pair-Encoding (BPE)) recognises 
# the difference between a token with or without a preceding blank, 
# and it generates a different ID.
#
# Reference: https://huggingface.co/transformers/model_doc/roberta.html#transformers.RobertaTokenizer
#   This tokenizer has been trained to treat spaces like parts of the tokens 
#   (a bit like sentencepiece) so a word will be encoded differently 
#   whether it is at the beginning of the sentence (without space) or not.

# script to obtain input_ids for stop words
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
stopword_tokenised_set = []
for token in stopword_list:
    # w/o a blank
    input_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    stopword_tokenised_set.extend(input_ids)

    # w/ a blank
    input_ids_emp = tokenizer(str(" " + token), add_special_tokens=False)["input_ids"]
    stopword_tokenised_set.extend(input_ids_emp)

    print(input_ids, tokenizer.decode(input_ids), 
          input_ids_emp, tokenizer.decode(input_ids_emp), token)

stopword_tokenised_set = set(stopword_tokenised_set)
print(stopword_tokenised_set, set(tokenizer.convert_ids_to_tokens(list(stopword_tokenised_set))))
"""

def count_stop_words_from_line(line: str):
    """Count number of words and its ratio in a segment."""
    num_stop_word = len([token for token in line.split(" ")
                                if token in stopword_list])
    ratio_stop_word = num_stop_word / len(line.split(' '))
    return num_stop_word, ratio_stop_word


def concatenate_sentences(sentences: list, max_length=512):
    """Utility function to concatenate sentences into str, while ensuring 
    the number of tokens should be less than 512."""
    chunk = ""
    total_num_tokens = 0

    for sentence in sentences:
        senlen = len(sentence.split(' '))
        if (total_num_tokens + senlen + 1) < max_length:
            if chunk != "":
                chunk += " " + sentence
                total_num_tokens += senlen + 1 # blank
            else:
                chunk = sentence
                total_num_tokens += senlen
        else:
            break
    
    return chunk, total_num_tokens


def preprocess_book(lines: List[str]):
    """Preprocess book corpus by each line.  
    Args:
        articles (List[str]): str list of lines.
    
    Returns:
        dict.

    References:
        https://github.com/huggingface/transformers/blob/9bdce3a4f91c6d53873582b0210e61c92bba8fd3/src/transformers/data/datasets/language_modeling.py#L19
        https://github.com/huggingface/transformers/blob/9bdce3a4f91c6d53873582b0210e61c92bba8fd3/src/transformers/data/data_collator.py#L120
        https://github.com/dhlee347/pytorchic-bert/blob/master/tokenization.py
        https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
    """    
    text_list = []
    overlen_lines = []
    num_stop_words = []
    ratio_stop_words = []
    seq_len_list = []

    chunk = ""
    total_num_tokens = 0
    for index, line in tenumerate(lines, disable=args.disable_tqdm):
        # We try to fill up the chunk such that the number of tokens in it is 
        # close to 512. Then, we use the chunk as an input sequence.

        num_tokens = len(line.split(' '))

        if (total_num_tokens + num_tokens) < 512:
            # add to chunk
            if chunk != "":
                chunk += " " + line
                total_num_tokens += num_tokens + 1
            else:
                chunk = line
                total_num_tokens += num_tokens

        else:
            # remove blanks
            text = chunk.strip()
            text = text.replace("\n", "")
            
            # add to lists
            text_list.append(text)
            
            if num_tokens < 512:
                # initialise again
                total_num_tokens = num_tokens
                chunk = line
            else:
                # over-length sample
                # put lists -> sentencize & use as a sample later
                overlen_lines.append(line)
                chunk = ""
                total_num_tokens = 0
    

    if overlen_lines != []:
        print("Preprocessing over-length samples in BookCorpus...")
        # split each str line into sentences
        sentencizer = Sentencizer()
        sentencised_lines = sentencizer(overlen_lines)

        for index, line in tenumerate(sentencised_lines, disable=args.disable_tqdm):
            # concatenate sentences with their maximum length 512
            text, total_num_tokens = concatenate_sentences(line, max_length=512)
            
            # remove blanks
            text = text.strip()
            text = text.replace("\n", "")
            if text == "" or len(text.split(" ")) <= 1:
                continue
            
            # add to lists
            text_list.append(text)

    return {"text": text_list}


def remove_wiki_info(example):
    """Remove unnecessary texts in the wikipedia corpus."""
    keywords = ("See also", "References", "Category")
    for keyword in keywords:
        index = example["text"].find(keyword)
        if index != -1:
            example["text"] = example["text"][:index]
    return example


def preprocess_wiki(articles: List[str]):
    """Preprocess wikipedia corpus by each article.  
    Args:
        articles (List[str]): str list of articles.
    
    Returns:
        dict.
    """
    text_list = []

    # split each str article into sentences
    print("Split wiki articles into sentences...")
    sentencizer = Sentencizer()
    sentencised_articles = sentencizer(articles, n_jobs=10)

    print("Generate Roberta samples...")
    for index, article in tenumerate(sentencised_articles, disable=args.disable_tqdm):
        # concatenate sentences with their maximum length 512
        text, total_num_tokens = concatenate_sentences(article, max_length=512)
        
        # remove blanks
        text = text.strip()
        text = text.replace("\n", "")
        if text == "" or len(text.split(" ")) <= 1:
            continue
        
        # add to lists
        text_list.append(text)
    
    return {"text": text_list}


def tokenize_example(tokenizer, example):
    """Tokenise batched examples using a pre-trained RoBERTa tokeniser."""
    tokenized_example = tokenizer(example['text'], max_length=512, padding="max_length", truncation="longest_first")

    return tokenized_example


def add_stop_word_masking_example(example):
    """Generate stop word masks (similar to attention masks)."""
    stop_word_mask = np.zeros_like(example["input_ids"])
    ones = np.ones_like(example["input_ids"])
    for target_id in stopword_id_list:
        stop_word_mask = np.where(example["input_ids"] == target_id, 
                                  ones,
                                  stop_word_mask)
    example["stop_word_mask"] = stop_word_mask

    return example


def mask_example(example, tokenizer, mask_prob=0.15):
    """Generate masked input sequences given batched samples."""
    # init
    labels = torch.from_numpy(example["input_ids"].copy()) # -> (bs, seq_len)
    probability_matrix = torch.full(labels.shape, mask_prob) # -> (bs, seq_len)

    # special token mask (incl. start/end, padding)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    # which token is going to be replaced?
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    example["masked_input_ids"] = labels.numpy()

    return example


def preprocess_dataset_for_roberta_pretraining(seed:int=1234,
                                         data_path=None, 
                                         save_on_disk:bool=True, 
                                         save_path=None):
    """Preprocess a dataset for pre-training
    Args:
        seed (int): random seed for shuffling
        data_path (str): the location of the processed dataset. This must be given
                         when `load_from_disk` is `True`.  
        save_on_disk (bool): whether to save a processed dataset on the disk.  
        save_path (str): where to save the processed dataset. This must be given
                         when `save_on_disk` is `True`.  
    
    References:
        https://huggingface.co/docs/datasets/splits.html
    """

    # setting up the path to save datasets
    if Path(save_path).exists() is False:
        Path(save_path).mkdir()
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    for ratio in range(0, 100, 10):
        # To avoid memory error, we preprocess datasets 10% each, then save it.

        # preprocess bookcorpus data
        book_dataset = load_dataset("bookcorpus", split=f"train[{ratio}%:{ratio+10}%]")
        print("Preprocessing BookCorpus...")
        book_dataset = preprocess_book(book_dataset["text"])

        # preprocess wiki data
        wiki_dataset = load_dataset("wikipedia", "20200501.en", split=f"train[{ratio}%:{ratio+10}%]")
        wiki_dataset.remove_columns_("title") # only keep the text based on the original BERT paper
        print("Removing unnecessary wiki data...")
        wiki_dataset = wiki_dataset.map(remove_wiki_info) # remove references etc.
        print("Preprocessing Wikipedia dataset...")
        wiki_dataset = preprocess_wiki(wiki_dataset["text"]) # make a sentence pair & labels

        # tokenisation
        print("Tokenising datasets...")
        book_dataset = datasets.Dataset.from_dict(book_dataset)
        wiki_dataset = datasets.Dataset.from_dict(wiki_dataset)
        bert_dataset = concatenate_datasets([book_dataset, wiki_dataset])
        bert_dataset = bert_dataset.shuffle(seed=seed)
        dataset = bert_dataset.map(lambda example: tokenize_example(tokenizer, example), 
                                   batched=True, batch_size=5000)

        """
        # this is needed to use `input_ids` in generating stop word masks
        # type should be ndarray: a `dict` of types like `(<class 'list'>, <class 'numpy.ndarray'>)`.
        dataset.set_format(type='np', columns=['input_ids', 'attention_mask', 
                                               'num_stop_word', 'ratio_stop_word'])
        
        # generate stop word masks
        print("Generating stop word masks...")
        dataset = dataset.map(lambda example: add_stop_word_masking_example(example),
                              batched=True, batch_size=64)
        
        # generate masked input sequences
        print("Generating masked input sequences...")
        dataset = dataset.map(lambda example: mask_example(example, tokenizer, args.mask_prob),
                              batched=True, batch_size=32)
        """
        # reset format
        dataset.set_format()
    
        if save_on_disk:
            print("Saving the processed data to disk...")
            if save_path is None:
                raise ValueError("Please give an appropriate path!")
            temp_save_path = Path(save_path) / str(ratio)
            if temp_save_path.exists() is False:
                temp_save_path.mkdir()
            dataset.save_to_disk(str(temp_save_path))
            print("Done!")


def main():
    # preprocess datasets
    preprocess_dataset_for_roberta_pretraining(save_path=args.path)


if __name__ == "__main__":
    main()
