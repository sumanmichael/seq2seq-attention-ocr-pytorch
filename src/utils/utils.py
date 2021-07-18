import collections
import re

import torch
import torch.nn as nn
from torch.autograd import Variable

SOS_TOKEN = 0  # special token for start of sentence
EOS_TOKEN = 1  # special token for end of sentence

class ConvertBetweenStringAndLabel(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict['SOS_TOKEN'] = SOS_TOKEN
        self.dict['EOS_TOKEN'] = EOS_TOKEN
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 2

    def encode(self, text):
        """
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.dict[item] if item in self.dict else 2 for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]
            max_length = max([len(x) for x in text])
            nb = len(text)
            targets = torch.ones(nb, max_length + 2) * 2
            for i in range(nb):
                targets[i][0] = 0
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = 1
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts

class Averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def load_data(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data) #TODO


def weights_init(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


def get_alphabet():
    with open('./data/devanagari-charset.txt', encoding="utf-8") as f:
        data = f.readlines()
        alphabet = [x.rstrip() for x in data]
        # alphabet += ' '
        return alphabet


def get_converted_word(decoder_outputs, get_prob=False):
    converter = ConvertBetweenStringAndLabel(get_alphabet())
    decoded_words = []
    if get_prob:
        prob = 1.0
    for decoder_output in decoder_outputs:
        _, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoded_words.append(converter.decode(ni))

        if get_prob:
            probs = torch.exp(decoder_output)
            prob *= probs[:, ni]

    if decoded_words[-1] == "EOS_TOKEN":
        decoded_words = decoded_words[:-1]

    words = ''.join(decoded_words)

    if get_prob:
        prob = prob.item()
        return words, prob
    else:
        return words


class digitIterator:
    def __init__(self, string, curr=0):
        self.string = string
        self.curr = curr
        self.length = len(string)
    def get_dig_len(self, increment=True):
        han4 = [2535,1040, 8221, 2985, 3006, 2792, 8377, 8226, 8220, 8216, 2798, 2965, 3585, 8211, 8212, 1041, 3594, 8204, 7386, 8205, 8211, 8217, 8220,8221]
        han2 = [93, 61, 42, 47, 58, 96, 63, 40, 37, 41, 59, 34, 43, 33,99]
        out = self.string
        i = self.curr
        length=-1
        if out[i:i+5]=='43251' and out[i:i+5]=='65279':
            length = 5
        elif out[i:i+2]=='23' or out[i:i+2]=='24' or int(out[i:i+4]) in han4: #10,82,29,30,27,83,35
            length = 4
        elif out[i:i+3] == '124' or out[i:i+3] == '183':
            length = 3
        elif out[i:i+2]<='96' and out[i:i+2]>='32':
            length = 2
        else:
            raise Exception
        if increment:
            self.curr+=length
        return length
    def get_next_unicode(self):
        return self.string[self.curr:self.curr+self.get_dig_len()]
    def get_ith_unicode(self,i):
        tempcurr = self.curr
        self.curr = 0
        for j in range(i):
            self.get_dig_len()
        tempunicode = self.get_next_unicode()
        self.curr = tempcurr
        return tempunicode
    def has_next(self):
        try:
            return self.curr + self.get_dig_len(False) <= self.length
        except :
            return False
    def get_next_char(self):
        return chr(int(self.get_next_unicode()))
    def get_ith_char(self,i):
        return chr(int(self.get_ith_unicode(i)))
    def get_ith_idxs(self,i):
        tempcurr = self.curr
        self.curr = 0
        for j in range(i):
            self.get_dig_len()
        currTuple = (self.curr, self.curr+self.get_dig_len())
        self.curr = tempcurr
        return currTuple
    def get_str(self):
        out=""
        while(self.has_next()):
            out += self.get_next_char()
        return out
    @staticmethod
    def getIdxsInfo(char_pred,subString):
        # this is with respect to the unicoded string and not the original char string
        mIdxs = digitIterator.getMatchingIndexes(char_pred,subString)
        if len(mIdxs)==0:
            return -1
        startEndInfo = []
        encoded_subString = ''.join([str(ord(x)) for x in subString])
        for idx in mIdxs:
            before_str = ''.join([str(ord(char_pred[x])) for x in range(idx)])
            startIdx = len(before_str)
            endIdx = startIdx+len(encoded_subString)
            startEndInfo.append(tuple([startIdx,endIdx]))
        return startEndInfo
    @staticmethod
    def getMatchingIndexes(char_pred,subString):
        return [m.start() for m in re.finditer(subString, char_pred)]