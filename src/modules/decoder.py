import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.modules.custom import Conv2dSamePadding


class AttentionDecoder(pl.LightningModule):
    """
        Use seq to seq model to modify the calculation method of attention weight
    """

    def __init__(self, attn_dec_hidden_size, enc_vec_size, enc_seq_length, target_embedding_size, target_vocab_size,
                 batch_size, LOAD_PATH=None):
        super(AttentionDecoder, self).__init__()
        self.enc_vec_size = enc_vec_size  # 512
        self.enc_seq_length = enc_seq_length  # 128
        self.batch_size = batch_size
        self.attn_dec_hidden_size = attn_dec_hidden_size  # 128
        self.target_embedding_size = target_embedding_size  # 10
        self.cell_input_size = attn_dec_hidden_size
        self.target_vocab_size = target_vocab_size

        self.conv_1x1 = Conv2dSamePadding(self.enc_vec_size, self.enc_vec_size, kernel_size=(1, 1), stride=1,
                                          bias=False)
        self.embedding = nn.Embedding(self.target_vocab_size, self.target_embedding_size)

        self.attention_projection = nn.Linear(self.enc_vec_size, self.enc_vec_size)

        self.input_projection = nn.Linear(self.target_embedding_size + self.enc_vec_size, self.cell_input_size)
        self.output_projection = nn.Linear(self.attn_dec_hidden_size + self.enc_vec_size, self.target_vocab_size)

        self.VT = nn.Linear(self.enc_vec_size, 1, bias=False)

        self.rnn = nn.LSTM(input_size=self.cell_input_size, hidden_size=self.attn_dec_hidden_size, num_layers=2)

        self.attention_weights_history = []
        self.encoder_output = None

        if LOAD_PATH:
            self.load_state_dict(torch.load(LOAD_PATH))

    def get_hidden(self):

        encoder_masks = []
        for i in range(128 + 1):
            encoder_masks.append(
                torch.tile(torch.tensor([[1.]]), [self.batch_size, 1])
            )

        encoder_inputs = [e * f for e, f in zip(self.encoder_output, encoder_masks[:128])]
        top_states = [torch.reshape(e, [-1, 1, 256 * 2]) for e in encoder_inputs]
        attention_states = torch.cat(top_states, 1)
        hidden = torch.reshape(attention_states, [-1, 128, 1, 512])
        return hidden

    def attention(self, state_flat):
        Wh_ht = self.attention_projection(state_flat)  #(l)
        Wh_ht = Wh_ht.view(-1, 1, 1, self.enc_vec_size)
                            # v~ (h_v = 512,t=128)
        conv_inp = self.encoder_output.unsqueeze(0)  # 1 128 4 512
        self.batch_size = state_flat.shape[0]


        # conv_inp = self.get_hidden()
        conv_inp = conv_inp.permute(2, 3, 0, 1)  # 4 512 1 128
        Wv_v = self.conv_1x1(conv_inp)  #(l?512,128)   # 4 512 1 128

        Wv_v = Wv_v.permute(0, 3, 2, 1) #4 128 1 512
        # Mat = Mat + Vec
        hy = Wv_v + Wh_ht  #(l?512,128)   #4 128 1 512
        hy = hy.squeeze(2)

        th = torch.tanh(hy)
        e = self.VT(th)  #(1,l?512) #(128)

        alpha = torch.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1) #4 1 128
        hidden = self.encoder_output.permute(1, 0, 2)
        context = torch.bmm(alpha, hidden)  #4 1 512
        # al_t = al_it for all i #(128)
        return context, alpha

    def forward(self, prev_output, attention_context, state):
        prev_symbol = torch.argmax(prev_output, dim=1)
        prev_emb = self.embedding(prev_symbol) #(B,ED) (?,10)
        inp_proj_inp = torch.cat((prev_emb, attention_context), dim=1)

        cell_inp = self.input_projection(inp_proj_inp)
        cell_inp = cell_inp.unsqueeze(0)
        cell_output, state = self.rnn(cell_inp, state)
        h, c = state
        state_flat = torch.cat((c[0], h[0], c[1], h[1]), dim=1)
        # TODO: COMPLETED
        attention_context, attention_weights = self.attention(state_flat)
        # moving to cpu to prevent bloating in GPU
        self.attention_weights_history.append(attention_weights.detach().to("cpu"))

        cell_output = cell_output.permute(1, 0, 2)
        out_proj_inp = torch.cat((cell_output, attention_context), dim=2)

        output = self.output_projection(out_proj_inp)
        output = output.squeeze(1)
        attention_context = attention_context.squeeze(1)

        return output, attention_context, state

    def set_encoder_output(self, encoder_output):
        self.encoder_output = encoder_output
