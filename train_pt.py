import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from src.modules.decoder import AttentionDecoder
from src.modules.encoder import Encoder
from src.utils import utils, dataset
from src.utils.utils import get_alphabet

from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex

alphabet = get_alphabet()

# define convert between string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 3


def train(train_loader, encoder, decoder, criterion, teach_forcing_prob=1):
    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

    # loss averager
    loss_avg = utils.Averager()
    num_classes = len(alphabet) + 3

    for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
        encoder_param.requires_grad = True
        decoder_param.requires_grad = True
    encoder.train()
    decoder.train()

    nvmlInit()

    for epoch in range(cfg.num_epochs):
        train_iter = iter(train_loader)

        for i in range(len(train_loader)):
            cpu_images, cpu_texts = train_iter.next()
            batch_size = cpu_images.size(0)

            encoder_outputs, state = encoder(cpu_images.cuda())
            state = utils.modify_state_for_tf_compat(state)
            decoder.set_encoder_output(encoder_outputs)
            attention_context = torch.zeros((batch_size, cfg.hidden_size * 2), device="cuda:0")

            target_variable = converter.encode(cpu_texts, "cpu")
            max_length = target_variable.shape[0]

            decoder_input = utils.get_one_hot(torch.tensor([utils.SOS_TOKEN] * batch_size, device="cuda:0"),
                                              num_classes)
            teach_forcing = True if random.random() > teach_forcing_prob else False

            target_variable = target_variable.cuda()

            loss = 0.0

            decoder_outputs = []

            for di in range(1, max_length):
                decoder_output, attention_context, state = decoder(decoder_input, attention_context, state)
                decoder_outputs.append(decoder_output)
                loss += criterion(decoder_output, target_variable[di])
                if teach_forcing and di != max_length - 1:
                    decoder_input = utils.get_one_hot(target_variable[di], num_classes)
                else:
                    _, topi = decoder_output.data.topk(1)
                    topi = topi.detach()
                    ni = topi.T[0]
                    decoder_input = utils.get_one_hot(ni, num_classes)

            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            loss_avg.add(loss)
            del loss
            encoder_optimizer.step()
            decoder_optimizer.step()

            if i % cf.log_interval == 0:
                info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
                print(
                    '[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4} | CUDA: {5:.2f}M/{6:.2f}M'.format(epoch, cfg.num_epochs,
                                                                                                 i, len(train_loader),
                                                                                                 loss_avg.val(),
                                                                                                 info.used / (
                                                                                                             1024 * 1024),
                                                                                                 info.total / (
                                                                                                             1024 * 1024)))
                loss_avg.reset()

        # save checkpoint
        # if epoch % cfg.save_interval == 0:
        #     torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(cfg.model, epoch))
        #     torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(cfg.model, epoch))


def main():
    if not os.path.exists(cfg.model):
        os.makedirs(cfg.model)

    # create train dataset
    train_dataset = dataset.TextLineDataset(root=cfg.train_list, transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=None, num_workers=int(cfg.num_workers),
        collate_fn=dataset.AlignCollate(img_height=cfg.img_height, img_width=cfg.img_width))

    # create test dataset
    test_dataset = dataset.TextLineDataset(root=cfg.eval_list, transform=dataset.ResizeNormalize(
                                                                                                    img_width=cfg.img_width, 
                                                                                                    img_height=cfg.img_height
                                                                                                )
                                                                                            )

    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,
                                              num_workers=int(cfg.num_workers))

    # create crnn/seq2seq/attention network
    encoder = Encoder(image_channels=1, enc_hidden_size=cfg.hidden_size)
    # for prediction of an indefinite long sequence
    attn_dec_hidden_size = 128
    enc_output_vec_size = 256 * 2
    enc_seq_length = 128
    target_embedding_size = 10
    batch_size = cfg.batch_size

    decoder = AttentionDecoder(
        attn_dec_hidden_size=attn_dec_hidden_size,
        enc_vec_size=enc_output_vec_size,
        enc_seq_length=enc_seq_length,
        target_embedding_size=target_embedding_size,
        target_vocab_size=num_classes,
        batch_size=batch_size
    )
    print(encoder)
    print(decoder)
    # encoder.apply(utils.weights_init)
    # decoder.apply(utils.weights_init)
    if cfg.encoder:
        print('loading pretrained encoder model from %s' % cfg.encoder)
        encoder.load_state_dict(torch.load(cfg.encoder))
    if cfg.decoder:
        print('loading pretrained encoder model from %s' % cfg.decoder)
        decoder.load_state_dict(torch.load(cfg.decoder))

    # create input tensor

    criterion = torch.nn.CrossEntropyLoss()

    assert torch.cuda.is_available(), "Please run \'train_pt.py\' script on nvidia cuda devices."
    encoder.cuda()
    decoder.cuda()
    criterion = criterion.cuda()

    # train crnn
    train(train_loader, encoder, decoder, criterion, teach_forcing_prob=cfg.teaching_forcing_prob)

    # do evaluation after training
    # evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)


if __name__ == "__main__":
    cudnn.benchmark = False
    # cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, help='path to train dataset list file')
    parser.add_argument('--eval_list', type=str, help='path to evalation dataset list file')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading num_workers')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--img_height', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--img_width', type=int, default=512, help='the width of the input image to network')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
    parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
    parser.add_argument('--model', default='./models/', help='Where to store samples and models')
    parser.add_argument('--random_sample', default=True, action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
    parser.add_argument('--max_width', type=int, default=129, help='the width of the feature map out from cnn')
    parser.add_argument('--save_interval', type=int, default=50, help='save for every ___ epochs')
    parser.add_argument('--log_interval', type=int, default=1000, help='log after every ___ iteration for each epoch.')
    cfg = parser.parse_args()
    print(cfg)

    main()
