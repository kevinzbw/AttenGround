import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import prepare_sequence
from const import START_TAG, END_TAG
import random

def func_attention(seq_f, img_f):
    """ 
    seq_f: batch, hidden, seq_len
    img_f: batch, hidden, img_h, img_w
    """
    batch_size, seq_len = seq_f.size(0), seq_f.size(2)
    img_h, img_w = img_f.size(2), img_f.size(3)
    img_len = img_h * img_w

    # batch, img_len, hidden
    img_f = img_f.view(batch_size, -1, img_len)
    img_f_T = torch.transpose(img_f, 1, 2).contiguous()

    # score: img_len * seq_len
    attn = torch.bmm(img_f_T, seq_f) # Eq. (7)
    attn = attn.view(batch_size*img_len, seq_len)
    attn = nn.Softmax()(attn)

    attn = attn.view(batch_size, img_len, seq_len)
    # score_T: seq_len * img_len
    attn_T = torch.transpose(attn, 1, 2).contiguous()

    # batch, hidden, img_len
    context_img = torch.bmm(seq_f, attn_T)
    context_img = context_img.view(batch_size, -1, img_h, img_w)
    # batch, hidden, seq_len
    context_seq = torch.bmm(img_f, attn) 

    attn_T = attn_T.view(batch_size, -1, img_h, img_w)

    return context_seq, context_img, attn_T

    
class AttenGround(nn.Module):

    def __init__(self, w_map, image_encoder, text_encoder, image_decoder, text_decoder, device):
        super().__init__()
        self.w_map = w_map
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_decoder = image_decoder
        self.text_decoder = text_decoder
        self.device = device

    def generate_seq(self, hidden_f, seq):
        batch_size = 1
        outputs = torch.zeros(seq.size(0), batch_size, self.text_encoder.vocab_size).to(self.device)
        
        hidden_f = hidden_f.view(hidden_f.size(1), -1)
        hidden = hidden_f.mean(1).view(2, 1, self.text_decoder.hidden_dim // 2)
        hidden = (hidden, hidden)
        
        input = prepare_sequence([START_TAG], self.w_map, self.device)
        
        for t in range(0, len(seq)):
            output, hidden = self.text_decoder(input, hidden)
            outputs[t] = output
            top1 = output.max(1)[1]
            if top1.item() == END_TAG:
                break
            input = top1
        
        outputs = outputs.view(outputs.size(0), -1)
        return outputs

    def forward(self, img, seq, teacher_forcing_ratio = 0.5):
        img_f = self.image_encoder(img)
        seq_f = self.text_encoder(seq)
        seq_f = seq_f[None, ]
        context_seq, context_img, _ = func_attention(seq_f=seq_f, img_f=img_f)
        context_seq, context_img, _ = func_attention(seq_f=context_seq, img_f=context_img)
        context_seq, context_img, _ = func_attention(seq_f=context_seq, img_f=context_img)
        context_seq, context_img, attn = func_attention(seq_f=context_seq, img_f=context_img)
        re_img = self.image_decoder(context_img)

        re_text = self.generate_seq(context_seq, seq)

        return re_img, re_text, attn

