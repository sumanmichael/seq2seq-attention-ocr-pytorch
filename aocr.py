import random

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import instantiate_class

from crnn.seq2seq import Encoder, Decoder
from src import utils, dataset


def get_alphabet():
    with open('./data/devanagari-charset.txt', encoding="utf-8") as f:
        data = f.readlines()
        alphabet = [x.rstrip() for x in data]
        alphabet += ' '
        return alphabet


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
            encoder_pth: str = None,
            decoder_pth: str = None,
            save_model_dir: str = None,
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
                    encoder_pth: path to encoder (to continue training)
                    decoder_pth: path to decoder (to continue training)
                    save_model_dir: Where to store samples and models
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
        self.encoder_pth = encoder_pth
        self.decoder_pth = decoder_pth
        self.save_model_dir = save_model_dir

        self.alphabet = get_alphabet()
        self.num_classes = len(self.alphabet) + 2  # len(alphabet) + SOS_TOKEN + EOS_TOKEN
        self.encoder = Encoder(channel_size=3, hidden_size=self.hidden_size).to(self.device)
        self.decoder = Decoder(hidden_size=self.hidden_size, output_size=self.num_classes, dropout_p=self.dropout_p,
                               max_length=self.max_enc_seq_len).to(self.device)

        if encoder_optimizer_args is None:
            encoder_optimizer_args = {
                "class_path":"torch.optim.Adam",
                "init_args":{
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

        self.image = torch.FloatTensor(self.batch_size, 3, self.img_height, self.img_width).to(self.device)

        self.encoder.apply(utils.weights_init)
        self.decoder.apply(utils.weights_init)

        if self.encoder_pth:
            print('loading pretrained encoder model from %s' % self.encoder_pth)
            self.encoder.load_state_dict(torch.load(self.encoder_pth))
        if self.decoder_pth:
            print('loading pretrained encoder model from %s' % self.decoder_pth)
            self.decoder.load_state_dict(torch.load(self.decoder_pth))

    def forward(self, cpu_images, cpu_texts, is_training=True):
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
                # TODO: is it same with dec_inp below?
                teach_forcing = True if random.random() > self.teaching_forcing_prob else False
            else:  # val
                teach_forcing = False

        else:  # test
            teach_forcing = False
            decoder_input = torch.zeros(1).long().to(self.device)

        decoder_outputs = []
        for di in range(1, max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                             encoder_outputs)
            decoder_outputs.append(decoder_output)
            if teach_forcing:
                decoder_input = target_variable[di]
            else:
                _, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
                # Stop in EOS even in training?
                if not is_training:
                    if ni == utils.EOS_TOKEN:
                        break

        return decoder_outputs

    def training_step(self, train_batch, batch_idx, optimizer_idx=None):
        cpu_images, cpu_texts = train_batch
        decoder_outputs = self.forward(cpu_images, cpu_texts, is_training=True)
        target_variable = self.converter.encode(cpu_texts).to(self.device)

        loss = 0.0
        for di, decoder_output in enumerate(decoder_outputs, 1):
            loss += self.criterion(decoder_output, target_variable[di])

        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx, optimizer_idx=None):
        cpu_images, cpu_texts = val_batch
        decoder_outputs = self.forward(cpu_images, cpu_texts, is_training=False)
        target_variable = self.converter.encode(cpu_texts).to(self.device)

        loss = 0.0

        for di, decoder_output in enumerate(decoder_outputs, 1):  # Last Dec
            loss += self.criterion(decoder_output, target_variable[di])

        # accuracy, ground_truth, pred_text = self.get_preds(cpu_texts, decoder_output)
        #
        # with open(self.pred_text_file, "a+", encoding="utf-8") as f:
        #     f.write(f'{accuracy}\t{pred_text}\t{ground_truth}\n')

        self.log('val_loss', loss, logger=True)
        return loss

    def get_preds(self, cpu_texts, decoder_output):
        target_variable = self.converter.encode(cpu_texts).to(self.device)
        decoded_label = [decoder_output.data.topk(1)[1].squeeze(1)]
        n_correct = 0
        for pred, target in zip(decoded_label, target_variable[1:, :]):
            if pred == target:
                n_correct += 1
        n_total = len(cpu_texts[0]) + 1
        pred_text = ''.join([self.converter.decode(ni) for ni in decoded_label])
        ground_truth = cpu_texts[0]  # Considering BatchSize = 1
        accuracy = n_correct / n_total
        return accuracy, ground_truth, pred_text

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
        self.img_height = img_height
        self.img_width = img_width
        self.num_workers = num_workers
        self.random_sampler = random_sampler

        self.train_dataset = None
        self.training_sampler = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_list:
            self.train_dataset = dataset.TextLineDataset(text_line_file=self.train_list, transform=None)
            self.training_sampler = dataset.RandomSequentialSampler(self.train_dataset, self.batch_size)

        if self.val_list:
            self.val_dataset = dataset.TextLineDataset(text_line_file=self.val_list,
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