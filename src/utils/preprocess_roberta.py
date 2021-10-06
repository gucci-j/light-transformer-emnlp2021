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

# all: lower, cap, up -> 816
stopword_id_list = {5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 
                    25, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 45, 
                    47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 
                    64, 66, 69, 70, 71, 75, 77, 79, 81, 83, 84, 85, 87, 88, 89, 
                    90, 91, 95, 96, 97, 98, 99, 100, 102, 103, 104, 106, 108, 
                    109, 110, 114, 117, 118, 119, 122, 123, 125, 127, 129, 133, 
                    136, 137, 139, 141, 142, 143, 144, 145, 147, 148, 149, 150, 
                    152, 154, 159, 160, 162, 163, 166, 167, 170, 172, 178, 179, 
                    182, 197, 208, 209, 211, 214, 215, 218, 219, 222, 223, 225, 
                    227, 230, 240, 241, 243, 250, 252, 255, 256, 258, 259, 260, 
                    261, 264, 268, 274, 276, 280, 281, 282, 286, 287, 289, 305, 
                    308, 318, 326, 345, 347, 348, 349, 350, 351, 354, 367, 368, 
                    370, 374, 381, 384, 385, 387, 399, 404, 405, 407, 415, 417, 
                    424, 429, 440, 448, 454, 456, 462, 463, 468, 473, 475, 487, 
                    495, 497, 500, 506, 519, 520, 524, 531, 548, 560, 565, 572, 
                    574, 579, 581, 590, 596, 597, 598, 605, 608, 616, 617, 627, 
                    630, 653, 658, 660, 673, 683, 700, 713, 717, 725, 757, 769, 
                    771, 832, 846, 854, 870, 874, 890, 894, 901, 938, 939, 965, 
                    970, 975, 978, 993, 995, 1003, 1017, 1021, 1065, 1075, 1106, 
                    1116, 1121, 1185, 1193, 1213, 1216, 1223, 1235, 1250, 1308, 
                    1322, 1336, 1398, 1405, 1409, 1423, 1456, 1464, 1481, 1491, 
                    1495, 1525, 1529, 1534, 1541, 1590, 1594, 1599, 1610, 1620, 
                    1691, 1694, 1705, 1708, 1711, 1723, 1729, 1740, 1779, 1793, 
                    1794, 1801, 1832, 1843, 1851, 1862, 1868, 1889, 1892, 1916, 
                    1918, 1936, 1941, 1944, 1979, 1990, 1993, 2025, 2068, 2076, 
                    2096, 2126, 2137, 2185, 2191, 2220, 2246, 2264, 2282, 2290, 
                    2306, 2336, 2362, 2387, 2409, 2444, 2486, 2492, 2512, 2515, 
                    2522, 2527, 2548, 2571, 2604, 2606, 2612, 2615, 2620, 2629, 
                    2661, 2667, 2688, 2709, 2747, 2765, 2796, 2808, 2847, 2864, 
                    2895, 3001, 3047, 3063, 3066, 3084, 3105, 3128, 3139, 3224, 
                    3243, 3263, 3326, 3355, 3394, 3411, 3450, 3486, 3559, 3609, 
                    3654, 3684, 3703, 3721, 3732, 3750, 3755, 3764, 3765, 3779, 
                    3785, 3842, 3908, 3935, 3945, 3955, 3972, 3990, 4014, 4028, 
                    4041, 4148, 4154, 4248, 4288, 4297, 4321, 4322, 4395, 4421, 
                    4528, 4546, 4581, 4584, 4629, 4668, 4688, 4783, 4820, 4979, 
                    4993, 4995, 5016, 5030, 5053, 5061, 5087, 5089, 5096, 5102, 
                    5121, 5216, 5273, 5330, 5365, 5488, 5495, 5499, 5525, 5570, 
                    5598, 5632, 5634, 5652, 5771, 5818, 5844, 5945, 5971, 5975, 
                    6006, 6015, 6025, 6179, 6209, 6222, 6233, 6278, 6319, 6323, 
                    6362, 6390, 6532, 6553, 6567, 6570, 6677, 6766, 6785, 6826, 
                    6834, 6871, 6968, 7025, 7029, 7199, 7254, 7301, 7325, 7333, 
                    7424, 7443, 7574, 7605, 7608, 7698, 7761, 7831, 7877, 7949, 
                    7981, 7982, 7997, 8041, 8105, 8127, 8155, 8167, 8228, 8275, 
                    8310, 8338, 8374, 8495, 8569, 8573, 8585, 8640, 8663, 8827, 
                    8856, 8877, 8901, 8981, 8987, 9006, 9009, 9012, 9064, 9112, 
                    9131, 9174, 9178, 9226, 9325, 9332, 9442, 9443, 9475, 9557, 
                    9673, 9690, 9847, 9918, 9962, 10010, 10105, 10127, 10145, 10237, 
                    10259, 10284, 10365, 10414, 10466, 10540, 10570, 10612, 10616, 
                    10643, 10652, 10669, 10705, 10777, 10786, 10836, 10859, 10926, 
                    10978, 11083, 11094, 11120, 11195, 11329, 11475, 11613, 11672, 
                    11773, 11913, 11938, 11974, 11990, 12050, 12135, 12178, 12186, 
                    12196, 12341, 12375, 12389, 12465, 12471, 12473, 12655, 12707, 
                    12724, 12861, 12925, 12948, 13040, 13331, 13354, 13367, 13387, 
                    13437, 13449, 13464, 13556, 13584, 13755, 13784, 13841, 13910, 
                    14003, 14010, 14079, 14229, 14257, 14279, 14314, 14356, 14447, 
                    14587, 14656, 14662, 14746, 14780, 14944, 15157, 15159, 15251, 
                    15446, 15605, 15852, 15952, 16005, 16536, 16625, 16837, 16897, 
                    16918, 16991, 17143, 17206, 17245, 17276, 17304, 17341, 17345, 
                    17346, 17485, 17526, 17717, 17754, 17779, 17781, 17853, 18075, 
                    18212, 18258, 18342, 18630, 18966, 19058, 19174, 19385, 19933, 
                    19935, 19981, 20060, 20235, 20311, 20319, 20328, 20343, 20685, 
                    20693, 21097, 21098, 21240, 21600, 21651, 21674, 21688, 21770, 
                    22008, 22062, 22268, 22392, 22886, 22985, 23033, 23142, 23201, 
                    23367, 23444, 23803, 24001, 24394, 24770, 24844, 24975, 24980, 
                    24989, 25017, 25073, 25101, 25133, 25252, 25447, 25518, 25565, 
                    25784, 25800, 26021, 26369, 26421, 26692, 26817, 26895, 27201, 
                    27291, 27331, 27409, 27510, 27698, 27768, 27789, 27847, 28073, 
                    28595, 28842, 29042, 29510, 29723, 29734, 29856, 29891, 29892, 
                    29902, 29919, 30012, 30104, 30354, 30484, 30518, 30532, 30536, 
                    30540, 30660, 30857, 30872, 30991, 31231, 31448, 31455, 31934, 
                    31940, 31954, 31963, 32054, 32060, 32112, 32114, 32251, 32260, 
                    32431, 32500, 32794, 32882, 32919, 33178, 33835, 34072, 34157, 
                    34290, 34318, 34790, 34912, 35299, 35369, 35420, 35457, 35507, 
                    35634, 35669, 36195, 36698, 36933, 37049, 37051, 37167, 37257, 
                    37350, 37460, 37492, 37666, 37782, 38114, 38158, 38374, 38421, 
                    38543, 38642, 38678, 39200, 39212, 39236, 39269, 39282, 39648, 
                    39789, 39973, 40181, 40262, 40375, 40839, 40921, 40930, 41026, 
                    41033, 41634, 41690, 41812, 41814, 42068, 42218, 42271, 42599, 
                    42737, 43041, 43216, 43570, 43822, 43845, 44299, 44471, 44513, 
                    44628, 44897, 45117, 45893, 45943, 46298, 47431}
"""
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
stopword_id_lists = []
for token in stopword_list:
    # w/o a blank
    input_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    stopword_id_lists.extend(input_ids)

    # w/ a blank
    input_ids_emp = tokenizer(str(" " + token), add_special_tokens=False)["input_ids"]
    stopword_id_lists.extend(input_ids_emp)

    # w/o a blank (capitalise)
    input_ids_cap = tokenizer(token.capitalize(), add_special_tokens=False)["input_ids"]
    stopword_id_lists.extend(input_ids_cap)

    # w/ a blank (capitalise)
    input_ids_cap_emp = tokenizer(str(" " + token.capitalize()), add_special_tokens=False)["input_ids"]
    stopword_id_lists.extend(input_ids_cap_emp)

    # w/o a blank (upper)
    input_ids_up = tokenizer(token.upper(), add_special_tokens=False)["input_ids"]
    stopword_id_lists.extend(input_ids_up)

    # w/ a blank (upper)
    input_ids_up_emp = tokenizer(str(" " + token.upper()), add_special_tokens=False)["input_ids"]
    stopword_id_lists.extend(input_ids_up_emp)

stopword_id_set = set(stopword_id_lists)
print(sorted(stopword_id_set), set(tokenizer.convert_ids_to_tokens(list(stopword_id_set))))
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
