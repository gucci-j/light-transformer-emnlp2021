from transformers import BatchEncoding, PreTrainedTokenizerBase
import torch
import re
import pickle

# for debugging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class DataCollatorForShuffledWordClassification:
    """
    Data collator used for shuffled word classification as pre-training.  
    This class assumes that samples are tokenised in advance.  

    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, shuffle_prob: float):
        self.tokenizer = tokenizer
        self.shuffle_prob = shuffle_prob


    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # Shuffle words and create word masks
        # shuffled_input_ids, shuffled_word_mask, active_indices = self.shuffle_tokens(batch["input_ids"])
        shuffled_input_ids, shuffled_word_mask = self.shuffle_tokens(batch["input_ids"])

        return {"input_ids": shuffled_input_ids, "attention_mask": batch["attention_mask"],
                "shuffled_word_mask": shuffled_word_mask}


    def shuffle_tokens(self, input_ids):
        """Prepare shuffled tokens inputs/masks."""
        # init
        labels = input_ids.clone()
        
        # create shuffled input_ids matrices
        shuffled_words = labels[:, torch.randperm(labels.size()[1])] # row-wise shuffle
        # We need to care about special tokens: start, end, pad, mask.
        # If shuffled words fall in these, they must be put back to their original tokens.
        # This might cause the case where a shuffled token is not actually a shuffled one,
        # but because the number of special tokens is small, it does not matter and might contribute to robustness.
        special_tokens_indices = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in shuffled_words.tolist()
        ]
        special_tokens_indices = torch.tensor(special_tokens_indices, dtype=torch.bool) # -> boolean indices
        shuffled_words[special_tokens_indices] = labels[special_tokens_indices]
        
        # which token is going to be shuffled?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full(labels.shape, self.shuffle_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        shuffled_indices = torch.bernoulli(probability_matrix).bool() # -> boolean indices

        # replace 15% of tokens with shuffled ones
        labels[shuffled_indices] = shuffled_words[shuffled_indices]

        return labels, shuffled_indices


class DataCollatorForShuffleRandomThreeWayClassification:
    """
    Data collator used for three-way shuffled/random/non-replaced classification as pre-training.
    This class assumes that samples are tokenised in advance.  

    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, manipulate_prob: float):
        self.tokenizer = tokenizer
        self.manipulate_prob = manipulate_prob


    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # Shuffle words and create word masks
        manipulated_input_ids, shuffle_random_mask = self.manipulate_tokens(batch["input_ids"])

        return {"input_ids": manipulated_input_ids, "attention_mask": batch["attention_mask"],
                "shuffle_random_mask": shuffle_random_mask}


    def manipulate_tokens(self, input_ids):
        """Prepare shuffled tokens inputs/masks."""
        # init
        manipulated_input_ids = input_ids.clone()
        shuffle_random_mask = torch.zeros_like(manipulated_input_ids)
        
        # create shuffled input_ids matrices
        shuffled_words = manipulated_input_ids[:, torch.randperm(manipulated_input_ids.size()[1])] # row-wise shuffle
        # We need to care about special tokens: start, end, pad, mask.
        # If shuffled words fall in these, they must be put back to their original tokens.
        # This might cause the case where a shuffled token is not actually a shuffled one,
        # but because the number of special tokens is small, it does not matter and might contribute to robustness.
        special_tokens_indices = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in shuffled_words.tolist()
        ]
        special_tokens_indices = torch.tensor(special_tokens_indices, dtype=torch.bool) # -> boolean indices
        shuffled_words[special_tokens_indices] = manipulated_input_ids[special_tokens_indices]
        
        # which token is going to be shuffled?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full(manipulated_input_ids.shape, self.manipulate_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in manipulated_input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        shuffled_indices = torch.bernoulli(probability_matrix).bool() # -> boolean indices
        manipulated_input_ids[shuffled_indices] = shuffled_words[shuffled_indices]
        shuffle_random_mask[shuffled_indices] = 1

        # replace some tokens with random ones
        # this should not override shuffled tokens.
        random_indices = torch.bernoulli(torch.full(manipulated_input_ids.shape, self.manipulate_prob)).bool() & ~shuffled_indices & ~special_tokens_mask
        random_words = torch.randint(len(self.tokenizer), manipulated_input_ids.shape, dtype=torch.long)
        manipulated_input_ids[random_indices] = random_words[random_indices]
        shuffle_random_mask[random_indices] = 2

        # We only compute loss on active tokens
        shuffle_random_mask[special_tokens_mask] = -100

        return manipulated_input_ids, shuffle_random_mask


class DataCollatorForRandomWordClassification:
    """
    Data collator used for random token classification as pre-training.
    This class assumes that samples are tokenised in advance.  

    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, random_prob: float):
        self.tokenizer = tokenizer
        self.random_prob = random_prob


    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # Shuffle words and create word masks
        manipulated_input_ids, random_word_mask = self.manipulate_tokens(batch["input_ids"])

        return {"input_ids": manipulated_input_ids, "attention_mask": batch["attention_mask"],
                "random_word_mask": random_word_mask}


    def manipulate_tokens(self, input_ids):
        """Prepare shuffled tokens inputs/masks."""
        # init
        manipulated_input_ids = input_ids.clone()
        random_word_mask = torch.zeros_like(manipulated_input_ids)
        
        # create replaced input_ids matrices
        replaced_words = torch.randint(len(self.tokenizer), manipulated_input_ids.shape, dtype=torch.long)
        # We need to care about special tokens: start, end, pad, mask.
        # If shuffled words fall in these, they must be put back to their original tokens.
        # This might cause the case where a shuffled token is not actually a shuffled one,
        # but because the number of special tokens is small, it does not matter and might contribute to robustness.
        special_tokens_indices = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in replaced_words.tolist()
        ]
        special_tokens_indices = torch.tensor(special_tokens_indices, dtype=torch.bool) # -> boolean indices
        replaced_words[special_tokens_indices] = manipulated_input_ids[special_tokens_indices]
        
        # which token is going to be replaced?
        # create special tokens' mask for original input_ids
        probability_matrix = torch.full(manipulated_input_ids.shape, self.random_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in manipulated_input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        replaced_indices = torch.bernoulli(probability_matrix).bool() # -> boolean indices
        manipulated_input_ids[replaced_indices] = replaced_words[replaced_indices]

        return manipulated_input_ids, replaced_indices


class DataCollatorForMaskedLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_prob: float=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob


    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # create inputs and targets for MLM
        inputs, labels = self.mask_tokens(batch["input_ids"])
        return {"input_ids": inputs, 
                "attention_mask": batch["attention_mask"], 
                "labels": labels}


    def mask_tokens(self, inputs: torch.Tensor):
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone() # this serves as a target

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_prob defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # By passing -100, nn.CrossEntropyLoss will ignore such labels when computing loss.
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


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

punctuation_id_list = {45056, 49153, 49154, 43012, 38917, 47110, 2055, 6, 4, 12, 
                       26638, 45072, 49170, 22, 28696, 18456, 49177, 49183, 22560, 
                       32801, 38947, 35, 36, 49189, 43048, 49193, 43, 49196, 47148,
                       41006, 49198, 49197, 49201, 47155, 36917, 12345, 47161, 
                       47162, 60, 49215, 49216, 43074, 30787, 68, 20551, 43080, 
                       72, 73, 28749, 49230, 41039, 49242, 43101, 45152, 49248, 
                       14434, 49255, 2153, 49258, 41066, 2156, 24681, 108, 30831, 
                       28784, 113, 111, 116, 2165, 45177, 16506, 49275, 49279, 128, 
                       49281, 131, 49283, 4234, 49291, 49293, 49296, 39058, 41110, 
                       47259, 49308, 49309, 49310, 49314, 49316, 49318, 41137, 49329, 
                       49333, 49338, 10431, 49346, 32965, 49358, 207, 49364, 49366, 
                       49371, 18653, 49374, 49380, 49384, 4332, 49389, 238, 49394, 
                       49410, 47365, 33031, 49419, 49420, 49423, 49424, 45333, 47385, 
                       49434, 43292, 49436, 49440, 49445, 43303, 8488, 43305, 49452, 
                       4397, 49453, 49455, 47408, 35122, 45364, 49463, 12606, 10559, 
                       45376, 322, 47426, 45381, 47429, 328, 31051, 45390, 49487, 
                       43344, 49489, 45393, 43353, 45405, 45406, 47457, 47460, 49509, 
                       359, 47463, 49515, 12651, 49518, 49519, 22896, 33137, 49521, 
                       49525, 49526, 31095, 49536, 37249, 24965, 29064, 43401, 49557, 
                       35227, 49563, 47517, 24992, 49570, 47529, 47539, 49599, 49604, 
                       39365, 31175, 27079, 49608, 6600, 49609, 16844, 49612, 49614, 
                       47567, 45518, 47570, 49625, 35290, 47579, 49629, 479, 480, 
                       47584, 482, 49632, 49639, 49643, 49655, 49659, 43521, 49666, 
                       47619, 47620, 12801, 41478, 49670, 27144, 49667, 49674, 49675, 
                       27148, 49681, 35347, 45587, 37398, 47639, 45592, 49688, 49690, 
                       1500, 49698, 49701, 49703, 47655, 6697, 31274, 47659, 43564, 
                       37421, 12846, 49710, 49712, 49713, 49721, 45627, 49727, 27203, 
                       49731, 49738, 49739, 31311, 41552, 37457, 49747, 49750, 27223, 
                       49755, 2652, 43613, 49761, 49763, 16998, 47720, 12905, 49774, 
                       49778, 43636, 49783, 49784, 49789, 39550, 45693, 49790, 640, 
                       49795, 646, 49799, 49798, 49800, 49803, 49806, 27282, 49812, 
                       49814, 39574, 49817, 47770, 47771, 49826, 21154, 49828, 49830, 
                       8871, 45737, 49836, 47789, 47793, 45751, 2744, 49849, 49852, 
                       49853, 49858, 49859, 35524, 47813, 4805, 41667, 49868, 49871, 
                       15057, 47826, 49882, 734, 10975, 49888, 45793, 49890, 4832, 
                       49893, 742, 4839, 49895, 43754, 45803, 49900, 49903, 49905, 
                       49908, 49909, 49910, 33525, 49915, 49918, 43775, 49921, 49923, 
                       49925, 41734, 19207, 37640, 37637, 49936, 49938, 787, 43796, 
                       31509, 29462, 49940, 21277, 49953, 49954, 43809, 45863, 49959, 
                       49962, 29483, 29482, 49964, 19246, 47919, 49969, 49972, 39732, 
                       49979, 49982, 43839, 49987, 39747, 17220, 45894, 838, 49991, 
                       31558, 49995, 845, 35661, 50000, 849, 19281, 50003, 50004, 
                       2901, 50007, 45912, 50012, 47965, 50014, 50015, 50016, 50017, 
                       50018, 50019, 50020, 50022, 50024, 50025, 41833, 50028, 33647, 
                       50031, 50037, 45946, 48004, 50053, 43912, 50061, 13198, 50065, 
                       21394, 50068, 45973, 50072, 48029, 48030, 50078, 50084, 48037, 
                       45994, 25522, 947, 955, 23500, 48077, 48081, 48082, 43988, 
                       48086, 41945, 11227, 13278, 23528, 50154, 50155, 48110, 1009, 
                       50161, 48119, 48124, 46077, 27645, 46082, 50179, 21509, 48134, 
                       25606, 50184, 50185, 50189, 1039, 50193, 48149, 50206, 44065, 
                       46117, 44082, 48182, 50236, 13373, 48188, 46142, 42053, 46150, 
                       48200, 48203, 46156, 50254, 50255, 17487, 46161, 48209, 44116, 
                       40021, 17495, 21594, 42078, 5214, 44128, 48229, 48232, 17516, 
                       17523, 25718, 15483, 35965, 48256, 44162, 27779, 27785, 48268, 
                       46225, 48273, 31897, 3226, 9376, 48289, 48292, 46250, 48298, 
                       46253, 48306, 3256, 1215, 44226, 19651, 21704, 48329, 48332, 
                       48336, 48342, 42199, 42202, 48347, 27868, 46303, 44259, 13540, 
                       48364, 48371, 29942, 48377, 15611, 36098, 44294, 46343, 48392, 
                       42248, 48395, 42254, 42255, 1297, 48404, 3358, 36137, 48433, 
                       46386, 7479, 42296, 38203, 48443, 40255, 48457, 48461, 21838, 
                       48462, 1358, 5457, 15698, 44371, 34133, 44374, 42326, 36185, 
                       48474, 44403, 44408, 48505, 48512, 48513, 40321, 44418, 46469, 
                       48520, 44431, 46479, 46481, 17809, 11665, 34199, 47096, 46495, 
                       44447, 38304, 7586, 30115, 48546, 48554, 44460, 23985, 48562, 
                       48565, 7606, 40389, 5579, 40398, 28114, 48601, 30171, 48610, 
                       44516, 48613, 48614, 48615, 46564, 46580, 48630, 48634, 48640, 
                       48651, 32269, 48654, 42514, 48660, 46613, 36380, 48669, 24095, 
                       48677, 13864, 48691, 1589, 48694, 1592, 46650, 1598, 36418, 
                       44612, 48709, 48712, 46671, 44626, 13909, 44629, 46679, 36440, 
                       48729, 1629, 46686, 42593, 38502, 48742, 1640, 42604, 48749, 
                       48755, 36467, 44660, 48759, 32376, 1666, 48771, 34437, 44688, 
                       48784, 44690, 48789, 42645, 42648, 48793, 48794, 42654, 44706, 
                       48803, 48805, 44717, 48817, 38581, 7862, 1721, 36538, 40635, 
                       48832, 22209, 48833, 48835, 14025, 48844, 48845, 44757, 18134, 
                       48855, 20186, 9957, 48872, 24303, 48880, 36592, 44793, 46844, 
                       48893, 48898, 48900, 48902, 42760, 48906, 48919, 42777, 44832, 
                       48935, 48936, 48937, 48948, 48950, 46904, 38713, 30529, 10068, 
                       46934, 48982, 30550, 46939, 10076, 48989, 3934, 48992, 46945, 
                       48999, 49000, 49007, 46961, 26487, 8061, 44926, 1917, 49024, 
                       36738, 10116, 8070, 14220, 44942, 49038, 24464, 46992, 16276, 
                       49051, 47006, 40862, 28578, 49069, 49070, 49071, 47033, 38844, 
                       49085, 47038, 49087, 49091, 49092, 49095, 24521, 49097, 47052, 
                       24524, 49104, 42964, 47075, 49123, 49128, 30697, 49130, 8174, 
                       49138, 26610, 49143, 36856, 49145, 43002, 43003, 49151}

digit_id_list = {32769, 32774, 40976, 40977, 32802, 40996, 24621, 41005, 32827, 
                 24646, 24649, 32857, 24675, 32869, 16491, 8301, 112, 41073, 16504,
                 41091, 132, 8325, 134, 24712, 41097, 41102, 32913, 41109, 151, 
                 155, 158, 32940, 176, 32954, 32957, 193, 195, 199, 32968, 204,
                 41165, 32972, 32976, 41168, 8403, 24789, 41179, 32994, 231, 49393,
                 245, 246, 24837, 262, 16662, 288, 290, 291, 41256, 41258, 33066,
                 33067, 41260, 306, 33075, 16692, 316, 41279, 321, 41285, 41295,
                 336, 41298, 16725, 41305, 33115, 33121, 8548, 41317, 361, 49514,
                 365, 49524, 379, 8572, 389, 41351, 49544, 41355, 398, 401, 406,
                 33176, 24990, 41374, 33185, 41383, 33192, 41401, 33213, 41409,
                 33225, 25034, 25037, 16846, 41424, 466, 33236, 25048, 25050, 16863,
                 49635, 33253, 41450, 41455, 501, 33269, 504, 33275, 508, 33279,
                 33287, 41485, 541, 16925, 545, 41512, 33325, 41518, 564, 8757,
                 33333, 41525, 570, 33342, 25155, 33352, 16970, 8783, 601, 16986,
                 25183, 33375, 612, 41573, 25208, 17024, 33410, 8835, 33413, 33416,
                 654, 41632, 33450, 33452, 17072, 41655, 41656, 698, 33467, 41663,
                 41665, 706, 41669, 41682, 33490, 727, 733, 25319, 8940, 753, 41714,
                 41717, 25345, 8963, 25352, 777, 33544, 8971, 49928, 49955, 41765,
                 33588, 41781, 820, 41782, 9029, 843, 844, 33611, 25423, 17235, 
                 25433, 33626, 41821, 41823, 25444, 33636, 883, 50036, 33657, 
                 50050, 41861, 9095, 33674, 25488, 41873, 33686, 50075, 25502, 33702, 
                 954, 25534, 33732, 25540, 33736, 971, 973, 33741, 974, 41946, 996, 
                 33772, 1014, 33783, 41977, 33787, 25631, 17445, 17472, 25664, 1092,
                 25669, 1096, 25676, 33871, 1105, 17500, 42080, 33888, 33891, 1125, 
                 9323, 1132, 42100, 33911, 25728, 33921, 33933, 42129, 42139, 25758, 
                 17573, 1191, 42154, 42161, 33973, 17594, 33981, 1225, 17616, 34009, 
                 1244, 42206, 34016, 34024, 34025, 34030, 42224, 34036, 17655, 9465, 
                 42249, 34067, 42262, 34071, 25884, 25895, 34088, 34094, 34099, 34109, 
                 17729, 42309, 25936, 1360, 1366, 25949, 34148, 42346, 25966, 25977, 
                 25983, 42367, 9633, 34212, 26022, 42412, 34221, 34224, 1466, 
                 42429, 34240, 34243, 42436, 34248, 34249, 42442, 9680, 42453, 1497, 
                 1510, 42478, 34288, 42482, 34295, 34309, 1549, 17936, 42513, 34322, 
                 34324, 1558, 17953, 1570, 34357, 34364, 42564, 26180, 26196, 34392, 
                 34401, 34411, 1646, 34414, 42610, 26227, 42617, 34429, 18069, 34454, 
                 26273, 26289, 1718, 34490, 26300, 34499, 34504, 34509, 1749, 26330, 
                 34523, 26332, 34531, 26340, 1764, 34539, 18159, 26372, 26373, 42770, 
                 1812, 42773, 1814, 18202, 1824, 34615, 10056, 1866, 42829, 34639, 
                 26447, 1878, 34660, 42853, 10088, 1898, 18283, 34676, 34682, 34684, 
                 1922, 26498, 26518, 34715, 34720, 42928, 18360, 1978, 34764, 26589, 
                 10206, 34782, 34781, 10213, 2022, 34795, 34801, 2036, 18427, 26629, 
                 34825, 26634, 10253, 18461, 2107, 34877, 43071, 43075, 34885, 34890, 
                 18508, 26703, 2146, 34916, 34921, 34923, 26738, 43130, 26748, 26754, 
                 26758, 26763, 10380, 34962, 34964, 34970, 34972, 43172, 18597, 34986,
                 34991, 34993, 26806, 18614, 18616, 35014, 35015, 2248, 35033, 2266, 
                 35041, 18661, 35048, 18673, 35058, 26866, 26868, 35061, 26871, 26873, 
                 26891, 43281, 35092, 43289, 26912, 10529, 2338, 35111, 26919, 35115, 
                 18733, 2357, 43319, 43327, 35142, 10569, 10572, 26957, 35153, 35154,
                 35155, 26972, 43366, 35178, 35183, 27003, 18817, 10626, 27011, 27012,
                 35208, 27018, 35211, 35225, 43420, 18844, 18847, 2464, 2466, 27044, 
                 2481, 35252, 35253, 27062, 35255, 18872, 2491, 10684, 35261, 35269, 
                 35278, 27089, 35284, 2517, 2518, 43479, 27097, 35300, 18918, 35303, 
                 35305, 43504, 35313, 2546, 2545, 10742, 27127, 35321, 27129, 35324, 
                 27142, 35337, 35340, 43532, 35343, 27153, 35348, 10775, 43545, 43554, 
                 27173, 35381, 19000, 2619, 35390, 43587, 27204, 35395, 2631, 35400,
                 35408, 35412, 35414, 43614, 35422, 2663, 43638, 27264, 19089, 35493,
                 27319, 35511, 19130, 35523, 2760, 19148, 35533, 2766, 35534, 35537, 35544,
                 27354, 35556, 43751, 43768, 27393, 2831, 43794, 27418, 35611, 2843, 35620,
                 27434, 35629, 35632, 11061, 27457, 2881, 27459, 35651, 19270, 2890,
                 35665, 43868, 2908, 27489, 43887, 2929, 27508, 35705, 2940, 35723, 
                 11151, 2965, 27546, 35739, 35741, 2983, 35752, 35757, 19376, 2993, 
                 35761, 35766, 35768, 35775, 3010, 35781, 35784, 27594, 35794, 35795, 
                 43993, 27612, 27623, 27624, 27637, 19446, 44023, 27639, 11265, 3079, 
                 35848, 3083, 35862, 35868, 3103, 3118, 19515, 3135, 11338, 27723, 
                 11343, 44113, 27732, 35925, 19547, 3170, 44133, 27758, 27761, 27765,
                 3191, 27784, 27786, 35980, 35992, 27806, 36011, 36012, 36016, 3248,
                 36022, 19641, 19644, 36039, 3272, 27855, 19671, 44248, 27873, 3305, 
                 27886, 36079, 27888, 36086, 36094, 3330, 36101, 44295, 3337, 27915,
                 36108, 36120, 36131, 27939, 3367, 44328, 36144, 11571, 44344, 3387,
                 27965, 36159, 27982, 27983, 3414, 3416, 27996, 36192, 19809, 36202,
                 36205, 28020, 36219, 36222, 36223, 36227, 28036, 44423, 36246, 11670,
                 36248, 3490, 3492, 3503, 28080, 3506, 28083, 44475, 36286, 11724,
                 36300, 11735, 44504, 3546, 3550, 28129, 3557, 44521, 36332, 3570,
                 44531, 36346, 36350, 28158, 28167, 28170, 28171, 3620, 44582, 36398,
                 36400, 36402, 28219, 28222, 28234, 36431, 36434, 3669, 3675, 36447,
                 28259, 36453, 28270, 20078, 3706, 36476, 3714, 36491, 20109, 44695,
                 36509, 28325, 28328, 36520, 36521, 3761, 44733, 28355, 3788, 3789,
                 36559, 36565, 36577, 28390, 44775, 36586, 3818, 36593, 28405, 36603, 
                 36610, 36611, 44808, 36621, 36632, 12060, 44835, 36643, 36647, 12087,
                 3897, 12096, 28482, 36678, 3913, 36681, 28490, 36688, 28497, 12112,
                 44880, 28503, 3933, 36711, 36716, 36720, 36721, 20352, 36745, 3982,
                 36751, 28558, 36754, 36766, 28577, 28581, 4006, 4013, 4015, 4017, 
                 4027, 44993, 4034, 36803, 28621, 36814, 28628, 36824, 4059, 36835,
                 4074, 28654, 28658, 28664, 36860, 36867, 36871, 36874, 4111, 36885,
                 12312, 28698, 36891, 4124, 12330, 36911, 28730, 36922, 4156, 45121,
                 36950, 4197, 28773, 20598, 28799, 36991, 28801, 36999, 37000, 28816,
                 37014, 37020, 28833, 28835, 28839, 4268, 45231, 37045, 28854, 37046,
                 4280, 4283, 28871, 37067, 20688, 45264, 37073, 12497, 20695, 37101,
                 28911, 37106, 37114, 37129, 20750, 28945, 37143, 28955, 37157, 4390,
                 28988, 37182, 37184, 4419, 37190, 37191, 20811, 4429, 4430, 37199,
                 45391, 4432, 37202, 4431, 37207, 12636, 29021, 29023, 29024, 12641,
                 45417, 29049, 45433, 37244, 37245, 4482, 29065, 37261, 37277, 37286,
                 37289, 29099, 4525, 37293, 37295, 37296, 45485, 37299, 4531, 37305,
                 12730, 29114, 4540, 37311, 37315, 37319, 29127, 12747, 37326, 37328,
                 37329, 29137, 4563, 37332, 29156, 37353, 45563, 20990, 29185, 29187,
                 37381, 29190, 37383, 21000, 45576, 37385, 29189, 21006, 37399, 37401,
                 37402, 37412, 29221, 37413, 37415, 4652, 37428, 29240, 45629, 21053,
                 4671, 29250, 37446, 21063, 37453, 21070, 29271, 29272, 4697, 29274,
                 21083, 37475, 4708, 37481, 29291, 4718, 12910, 37494, 29304, 37499,
                 21118, 29312, 21123, 21138, 45716, 37524, 37534, 4772, 37543, 29355,
                 29362, 29373, 4801, 21190, 37593, 37595, 37598, 45792, 13025, 29411,
                 37613, 29425, 21234, 29431, 37628, 45820, 37631, 37634, 29443, 29451,
                 37645, 4893, 37663, 37664, 37665, 37674, 37676, 37708, 4956, 29534, 
                 13156, 37736, 21353, 29546, 21358, 37748, 4981, 21369, 37753, 45947,
                 29561, 29567, 29568, 29572, 4999, 29576, 37767, 37770, 21397, 21403,
                 29600, 37793, 37795, 37797, 37810, 37811, 5046, 21436, 37821, 29630,
                 37822, 21448, 13259, 5067, 37839, 21458, 37853, 29670, 29672, 37866,
                 37867, 37871, 5114, 37887, 37892, 37900, 37901, 37905, 5138, 37911,
                 29721, 5155, 21540, 37932, 5169, 29748, 21566, 46152, 37964, 5208,
                 21598, 29794, 13411, 5220, 37988, 29810, 5241, 5243, 38012, 29821,
                 38014, 13442, 38020, 38027, 38037, 13466, 13470, 38049, 38051, 38063,
                 38071, 38075, 38080, 29899, 29903, 21713, 38097, 5334, 29911, 5339,
                 5352, 29930, 5356, 38127, 29936, 38129, 38138, 38139, 29949, 38143,
                 5379, 38149, 38151, 29961, 38156, 29969, 38161, 29980, 38173, 30003,
                 30011, 30024, 5449, 38220, 38222, 38223, 30042, 38235, 30043, 30050,
                 13668, 5479, 38249, 13674, 21867, 21868, 30069, 21891, 30085, 30092,
                 21903, 13726, 30111, 38305, 5545, 21933, 38318, 5549, 5553, 38324,
                 30137, 21947, 21948, 38348, 38349, 30168, 38362, 5595, 5594, 5606,
                 5607, 30190, 30205, 30207, 38405, 38413, 38415, 30224, 38425, 5659,
                 38427, 46621, 5663, 38433, 30242, 38434, 38440, 38444, 5677, 13872,
                 38456, 22078, 5714, 38485, 38500, 5735, 38504, 38508, 38510, 30320,
                 38514, 13950, 22145, 22146, 13955, 38530, 30342, 5773, 30359, 38554,
                 38578, 38588, 38590, 30398, 38599, 38601, 30419, 38611, 38613, 38616,
                 22236, 38620, 38633, 30442, 22260, 46847, 38671, 38679, 30487, 5913,
                 30488, 22300, 30509, 14130, 38708, 46901, 38714, 38715, 38717, 30527, 
                 38719, 46920, 5962, 30548, 38749, 5987, 38759, 38765, 46959, 38770,
                 22388, 30583, 14200, 30586, 38781, 38785, 46983, 22410, 22418, 38803,
                 30628, 38828, 38848, 38852, 30662, 30664, 30676, 38878, 30687, 6115,
                 38886, 38888, 6121, 38893, 30703, 38896, 38900, 30716, 38911, 38925,
                 38930, 30739, 6164, 30741, 38944, 38949, 38960, 6193, 30775, 6200,
                 38974, 22599, 38992, 14420, 38997, 38999, 6232, 22619, 14428, 30818,
                 39015, 39026, 39032, 30858, 39055, 39060, 39062, 14488, 30875, 39071,
                 30881, 14512, 47281, 14515, 14517, 39094, 39101, 39103, 22726, 47303,
                 14544, 30928, 30930, 39123, 39125, 6361, 22748, 39134, 39138, 39141,
                 30950, 39147, 47342, 30959, 14576, 39155, 39157, 14586, 30973, 22782,
                 6405, 39179, 30995, 39190, 39195, 39196, 22814, 31023, 31025, 6468,
                 6478, 31057, 39252, 31060, 22871, 39259, 39263, 39265, 31080, 47477, 
                 6521, 31099, 39293, 47489, 14722, 31105, 39311, 22928, 6551, 31132, 
                 39327, 6560, 14753, 14757, 39335, 31145, 39337, 39344, 22960, 39345, 
                 39350, 39352, 39356, 39372, 31181, 39373, 22991, 31184, 31185, 39382, 
                 31191, 6617, 23006, 39392, 39402, 39405, 31218, 31219, 39413, 31222, 
                 23041, 6657, 6668, 39440, 14873, 31262, 14881, 31271, 31276, 6705, 
                 6708, 39476, 23102, 31297, 31300, 39505, 39507, 39517, 6750, 23137,
                 31330, 39526, 31335, 39543, 31353, 39549, 39558, 31366, 6791, 39570,
                 39575, 15002, 15007, 31394, 39590, 39593, 31412, 23221, 31416, 31419,
                 39613, 39617, 39619, 39624, 39629, 39642, 15077, 23272, 39659, 39664,
                 39669, 31487, 31488, 39681, 31490, 23301, 39688, 39711, 15140, 23334,
                 23342, 39738, 31546, 39739, 31551, 15171, 31556, 39755, 31573, 31576,
                 7004, 39785, 39786, 39794, 23417, 39811, 39819, 31633, 39827, 23443,
                 31640, 39846, 39850, 15274, 31663, 39864, 15294, 15314, 23513, 39898,
                 31707, 39918, 23537, 31731, 7158, 31738, 31740, 39935, 39947, 15386,
                 23580, 31773, 39970, 31782, 40017, 31830, 40027, 15452, 40030, 40044,
                 23672, 40056, 31866, 40065, 31877, 23688, 40082, 23703, 40092, 40093,
                 31907, 15524, 31920, 40125, 40141, 31956, 40149, 7383, 40156, 23777,
                 23778, 40178, 40184, 7417, 31994, 31998, 40193, 32004, 40197, 40199,
                 40204, 23821, 40212, 32036, 40235, 40250, 40260, 40263, 32071, 32072,
                 40278, 32088, 32089, 40284, 23902, 40294, 32103, 7528, 40296, 48491,
                 40299, 23942, 15751, 32145, 7571, 40349, 32160, 40355, 7589, 15786,
                 40366, 40379, 40384, 40391, 32230, 24050, 48636, 7694, 40464, 24083,
                 15891, 15900, 24108, 32301, 32303, 32306, 32307, 24133, 48711, 40521,
                 32329, 24137, 40525, 40528, 15966, 32360, 7784, 24174, 40558, 40567,
                 40569, 32380, 40572, 40583, 40588, 40598, 16041, 40625, 40631, 40636, 
                 40652, 40663, 40670, 32486, 32490, 48876, 40685, 40690, 32498, 40709, 
                 40714, 40717, 24334, 40721, 7953, 32532, 40729, 16157, 32542, 7969, 
                 24355, 40740, 40761, 7994, 24378, 40766, 8008, 8017, 24402, 24404, 
                 24406, 40793, 16218, 32607, 40801, 32613, 40807, 32620, 32624, 
                 16242, 40819, 24436, 8060, 32638, 40830, 40832, 32646, 40840, 24462, 
                 40847, 40849, 16273, 40851, 49044, 8101, 40870, 16295, 16310, 40892, 
                 32701, 16316, 40893, 32706, 40903, 40912, 8148, 16344, 8157, 24543, 
                 8162, 40935, 24554, 8176, 40944, 40947, 32760, 40954, 40958}


def get_stop_word_mask(val):
    return list(map(lambda x: 1 if x in stopword_id_list else 0, val))


def get_digit_mask(val):
    return list(map(lambda x: 1 if x in digit_id_list else 0, val))


def get_punctuation_mask(val):
    return list(map(lambda x: 1 if x in punctuation_id_list else 0, val))


class DataCollatorForFourWayTokenTypeClassification:
    """
    Data collator used for four-way classification of a stop, content, number, 
    or punctuation token.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_prob: float):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
    

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # mask tokens and create word masks (labels)
        masked_input_ids, masked_word_labels = self.mask_tokens(batch["input_ids"])

        return {"input_ids": masked_input_ids, "attention_mask": batch["attention_mask"],
                "masked_word_labels": masked_word_labels}


    def mask_tokens(self, input_ids):
        """Prepare masked tokens and their labels."""
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # init
        masked_word_labels = torch.zeros_like(input_ids)

        # create stop word labels
        stop_word_mask = [
            get_stop_word_mask(val) for val in input_ids.tolist()
        ]
        stop_word_mask = torch.tensor(stop_word_mask, dtype=torch.bool)
        masked_word_labels[stop_word_mask] = 1

        # create digit labels
        digit_mask = [
            get_digit_mask(val) for val in input_ids.tolist()
        ]
        digit_mask = torch.tensor(digit_mask, dtype=torch.bool)
        masked_word_labels[digit_mask] = 2

        # create punctuation labels
        punctuation_mask = [
            get_punctuation_mask(val) for val in input_ids.tolist()
        ]
        punctuation_mask = torch.tensor(punctuation_mask, dtype=torch.bool)
        masked_word_labels[punctuation_mask] = 3
        
        # create a mask
        probability_matrix = torch.full(masked_word_labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_word_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # replace tokens with [MASK]
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return input_ids, masked_word_labels


class DataCollatorForFirstCharPrediction:
    """Data collator used for first character prediction."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_prob: float):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        
        # create label reference
        vocab_dict = tokenizer.get_vocab()
        self.id_to_label = {}
        char_to_label = {
            "a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9,
            "k":10, "l":11, "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18,
            "t":19, "u":20, "v":21, "w":22, "x":23, "y":24, "z":25
        }
        for token, token_id in vocab_dict.items():
            if token_id in punctuation_id_list:
                self.id_to_label[token_id] = 26
                continue
            
            if token_id in digit_id_list:
                self.id_to_label[token_id] = 27
                continue
            
            if token[0] == "Ġ" and len(token) != 1:
                label = char_to_label.get(token[1].lower(), None)
                if label is None:
                    self.id_to_label[token_id] = 28 # exceptional case
                else:
                    self.id_to_label[token_id] = label
            else:
                label = char_to_label.get(token[0].lower(), None)
                if label is None:
                    self.id_to_label[token_id] = 28 # exceptional case
                else:
                    self.id_to_label[token_id] = label
        

    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # mask tokens and create word masks (labels)
        masked_input_ids, masked_word_labels = self.mask_tokens(batch["input_ids"])

        return {"input_ids": masked_input_ids, "attention_mask": batch["attention_mask"],
                "masked_word_labels": masked_word_labels}


    def mask_tokens(self, input_ids):
        """Prepare masked tokens and their labels."""
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # init
        masked_word_labels = torch.zeros_like(input_ids)
        input_id_sets = set(input_ids.view(-1).tolist())

        # create labels
        for input_id in input_id_sets:
            masked_word_labels[input_ids == input_id] = self.id_to_label[input_id]
        
        # create a mask
        probability_matrix = torch.full(masked_word_labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_word_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # replace tokens with [MASK]
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return input_ids, masked_word_labels

