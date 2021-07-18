import random

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import instantiate_class

from src.seq2seq import Encoder, Decoder
from src.utils import utils, dataset
from src.utils.metrics import WER, CER


class OCR(pl.LightningModule):
    def __init__(
            self,
            hidden_size: int = 256,
            max_enc_seq_len: int = 129,
            batch_size: int = 4,
            img_height: int = 32,
            img_width: int = 512,
            teaching_forcing_prob: float = 0.5,
            learning_rate: float = 0.0001,
            dropout_p: float = 0.1,
            output_pred_path: str = 'output.txt',
            decoder_optimizer_args: dict = None,
            encoder_optimizer_args: dict = None,

    ):
        """OCR model

                Args:
                    hidden_size: size of the lstm hidden state
                    max_enc_seq_len: the width of the feature map out from cnn
                    batch_size: input batch size
                    img_height: the height of the input image to network
                    img_width: the width of the input image to network
                    teaching_forcing_prob: percentage of samples to apply teach forcing
                    learning_rate: learning_rate
                    dropout_p: Dropout probability in Decoder Dropout layer


        """
        super(OCR, self).__init__()

        self.hidden_size = hidden_size
        self.max_enc_seq_len = max_enc_seq_len
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.teaching_forcing_prob = teaching_forcing_prob
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p
        self.output_pred_path = output_pred_path

        self.alphabet = utils.get_alphabet()
        self.num_classes = len(self.alphabet) + 2  # len(alphabet) + SOS_TOKEN + EOS_TOKEN
        self.encoder = Encoder(channel_size=3, hidden_size=self.hidden_size).to(self.device)
        self.decoder = Decoder(hidden_size=self.hidden_size, output_size=self.num_classes, dropout_p=self.dropout_p,
                               max_length=self.max_enc_seq_len).to(self.device)



        if encoder_optimizer_args is None:
            encoder_optimizer_args = {
                "class_path": "torch.optim.Adam",
                "init_args": {
                    "lr": 0.0001
                }
            }
        if decoder_optimizer_args is None:
            decoder_optimizer_args = {
                "class_path": "torch.optim.Adam",
                "init_args": {
                    "lr": 0.0001
                }
            }

        self.encoder_optimizer_args = encoder_optimizer_args
        self.decoder_optimizer_args = decoder_optimizer_args

        self.criterion = torch.nn.NLLLoss()
        self.converter = utils.ConvertBetweenStringAndLabel(self.alphabet)
        self.wer = WER()
        self.cer = CER()

        self.image = torch.FloatTensor(self.batch_size, 3, self.img_height, self.img_width).to(self.device)

        self.encoder.apply(utils.weights_init)
        self.decoder.apply(utils.weights_init)

        self.encoder.load_state_dict(torch.load("data/encoder_1000.pth"))
        self.decoder.load_state_dict(torch.load("data/decoder_1000.pth"))

    def forward(self, cpu_images, cpu_texts, is_training=True, return_attentions=False):
        utils.load_data(self.image, cpu_images)
        self.image = self.image.to(self.device)
        batch_size = cpu_images.shape[0]
        encoder_outputs = self.encoder(self.image)
        decoder_hidden = self.decoder.initHidden(batch_size).to(self.device)
        max_length = self.max_enc_seq_len

        if cpu_texts is not None:  # train / val
            target_variable = self.converter.encode(cpu_texts).to(self.device)
            max_length = target_variable.shape[0]
            decoder_input = target_variable[utils.SOS_TOKEN].to(self.device)

            if is_training:  # train
                teach_forcing = True if random.random() > self.teaching_forcing_prob else False
            else:  # val
                teach_forcing = False

        else:  # test
            teach_forcing = False
            decoder_input = torch.zeros(1).long().to(self.device)

        decoder_outputs = []
        attention_matrix = []

        for di in range(1, max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                             encoder_outputs)
            decoder_outputs.append(decoder_output)
            if teach_forcing and di != max_length-1:
                decoder_input = target_variable[di]
            else:
                _, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
                # Stop in EOS even in training?
                if not is_training:
                    if ni == utils.EOS_TOKEN:
                        break
            if return_attentions:
                attention_matrix.append(decoder_attention)

        if return_attentions:
            attention_matrix = torch.stack(attention_matrix).permute(1, 0, 2).unsqueeze(0)  # [1,D,E]
        return decoder_outputs, attention_matrix

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        cpu_images, cpu_texts = train_batch
        decoder_outputs, _ = self.forward(cpu_images, cpu_texts, is_training=True, return_attentions=False)
        target_variable = self.converter.encode(cpu_texts).to(self.device)

        loss = 0.0
        for di, decoder_output in enumerate(decoder_outputs, 1):
            loss += self.criterion(decoder_output, target_variable[di])

        metrics = {
            'loss': loss,
            # 'train_wer': self.wer(decoder_outputs, target_variable)   # batch WER?
        }
        self.log_dict(metrics, logger=True)
        return metrics

    def validation_step(self, val_batch, batch_idx, optimizer_idx=None):
        cpu_images, cpu_texts = val_batch
        decoder_outputs, _ = self.forward(cpu_images, cpu_texts, is_training=False, return_attentions=False)
        target_variable = self.converter.encode(cpu_texts).to(self.device)

        loss = 0.0

        for di, decoder_output in enumerate(decoder_outputs, 1):  # Last Dec
            loss += self.criterion(decoder_output, target_variable[di])

        log_dict = {
            'val_loss': loss,
            'val_wer': self.wer(decoder_outputs, target_variable),
            'val_cer': self.cer(decoder_outputs, target_variable)
        }
        self.log_dict(log_dict, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx, optimizer_idx=None):
        cpu_images, _ = test_batch
        decoder_outputs, attention_matrix = self.forward(cpu_images, None, is_training=False, return_attentions=True)
        return utils.get_converted_word(decoder_outputs), attention_matrix

    # def configure_optimizers(self):
    #     encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
    #     decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
    #
    #     return encoder_optimizer, decoder_optimizer

    def configure_optimizers(self):
        encoder_optimizer = instantiate_class(self.encoder.parameters(), self.encoder_optimizer_args)
        decoder_optimizer = instantiate_class(self.decoder.parameters(), self.decoder_optimizer_args)

        return encoder_optimizer, decoder_optimizer


class OCRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_list: str = None,
            val_list: str = None,
            test_list: str = None,
            img_height: int = 32,
            img_width: int = 512,
            num_workers: int = 2,
            batch_size: int = 4,
            random_sampler: bool = True
    ):
        """

        Args:
            train_list: path to train dataset list file
            val_list: path to validation dataset list file
            test_list: path to test dataset list file
            img_height: the height of the input image to network
            img_width: the width of the input image to network
            num_workers: number of data loading num_workers
            batch_size: input batch size
            random_sampler: whether to sample the dataset with random sampler
        """
        super(OCRDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.img_height = img_height
        self.img_width = img_width
        self.num_workers = num_workers
        self.random_sampler = random_sampler

        self.train_dataset = None
        self.training_sampler = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_list:
            self.train_dataset = dataset.TextLineDataset(text_line_file=self.train_list, transform=None)
            self.training_sampler = dataset.RandomSequentialSampler(self.train_dataset, self.batch_size)

        if self.val_list:
            self.val_dataset = dataset.TextLineDataset(text_line_file=self.val_list,
                                                       transform=dataset.ResizeNormalize(img_width=self.img_width,
                                                                                         img_height=self.img_height))

        if self.test_list:
            self.test_dataset = dataset.TextLineDataset(text_line_file=self.test_list,
                                                        transform=dataset.ResizeNormalize(img_width=self.img_width,
                                                                                          img_height=self.img_height))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.training_sampler,
            num_workers=int(self.num_workers),
            collate_fn=dataset.AlignCollate(img_height=self.img_height, img_width=self.img_width))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, batch_size=1,
                                           num_workers=int(self.num_workers))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, batch_size=1,
                                           num_workers=int(self.num_workers))
