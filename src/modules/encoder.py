import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.modules.custom import Conv2dSamePadding, BatchNorm2d


class Encoder(pl.LightningModule):
    '''
        CNN+BiLstm does feature extraction
    '''

    def __init__(self, image_channels, enc_hidden_size, LOAD_PATH=None):
        super(Encoder, self).__init__()


        self.cnn = nn.Sequential(  # 1x32x512    # CxHxW
            Conv2dSamePadding(image_channels, 64, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x32x256
            Conv2dSamePadding(64, 128, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8x128
            Conv2dSamePadding(128, 256, 3, 1, 0, bias=False), BatchNorm2d(256), nn.ReLU(True),  # 256x8x128
            Conv2dSamePadding(256, 256, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x128
            Conv2dSamePadding(256, 512, 3, 1, 0, bias=False), BatchNorm2d(512), nn.ReLU(True),  # 512x4x128
            Conv2dSamePadding(512, 512, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x128
            Conv2dSamePadding(512, 512, 2, 1, 0, bias=False), BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # nn.Dropout(p=0.5)
        )  # 512x1x128 CxHxW
        # TODO: Dropout
        self.rnn = nn.LSTM(input_size=512, hidden_size=enc_hidden_size, bidirectional=True)

        if LOAD_PATH is not None:
            self.load_state_dict(torch.load(LOAD_PATH))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        # for i,l in enumerate(self.cnn):
        #     input = l(input)
        #     # if(i==21):
        #     #     print(l)
        #     #     print(input.permute(0,2,3,1)[0][0][0])
        # conv = input
        # print(conv.permute(0,2,3,1)[0][0][0])

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"

        conv = conv.squeeze(2)  #(b,c,w)
        # (seq, bat, chan) <-> (w, b, c)

        rnn_inp = conv.permute(2,0,1)
        encoder_outputs, (h_n, c_n) = self.rnn(rnn_inp)
        return encoder_outputs, (h_n, c_n)