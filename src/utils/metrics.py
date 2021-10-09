import torch
from rapidfuzz.string_metric import levenshtein
from torchmetrics import Metric

from src.utils import utils


class WER(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(WER, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.alphabet = utils.get_alphabet()

        self.add_state("wer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_words", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        pred_str = utils.get_converted_word(preds)
        target_str = "".join([self.alphabet[t.item()-3] for t in target[1:]])     # -3 is to exclude SOS EOS OOV

        self.wer += self.wer_calc(pred_str, target_str)
        self.n_words += len(target_str.split())

    def compute(self):
        wer = float(self.wer) / self.n_words
        return wer.item() * 100

    def wer_calc(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return levenshtein(''.join(w1), ''.join(w2))


class CER(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(CER, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.alphabet = utils.get_alphabet()

        self.add_state("cer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_chars", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        pred_str = utils.get_converted_word(preds)
        target_str = "".join([self.alphabet[t.item()-3] for t in target[1:]])     # -3 is to exclude SOS EOS OOV

        self.cer += self.cer_calc(pred_str, target_str)
        self.n_chars += len(target_str.replace(' ', ''))

    def compute(self):
        cer = float(self.cer) / self.n_chars
        return cer.item() * 100

    def cer_calc(self, s1, s2):
        s1, s2 = s1.replace(' ', ''), s2.replace(' ', '')
        return levenshtein(s1, s2)
