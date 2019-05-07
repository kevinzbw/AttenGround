import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


from dataset.dataloder import Flickr30k, Flickr30k_stats
from model.image_encoder import ImageEncoder
from model.image_decoder import ImageDecoder
from model.text_encoder import TextEncoder
from model.text_decoder import TextDecoder
from model.model import AttenGround

from utils import prepare_sequence

from const import START_TAG, END_TAG

root = './flickr30k_images'
ann_root = './flickr30k_entities'

TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
VOCABULARY_SIZE = 2000
EMBED_DIM = 100
ATTEN_DIM = 200
IMG_SIZE = 299
NUM_EPOCH = 50
LR = 0.01
WEIGHT_DECAY=5e-4
LOG_FREQ = 200
PATH = './ckpt.pth'

word_embedding_file = "./glove.6B.100d.txt"

# w2embed = {}
# with open(word_embedding_file, 'r') as f:
#     for line in f.readlines():
#         sp = line.strip().split(' ')
#         w2embed[sp[0]] = np.array([float(n) for n in sp[1:]])
# for w in w_map.keys():
#     if w not in w2embed:
#         print(w)

transform_train = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1)),
])

transform_test = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1)),
])

train_dataset = Flickr30k(root=root, ann_root=ann_root, train=True, vocab_size=VOCABULARY_SIZE, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

w_map = train_dataset.get_w_map()

test_dataset = Flickr30k(root=root, ann_root=ann_root, train=False, vocab_size=VOCABULARY_SIZE, w_map=w_map, transform=transform_test)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

device = 'cuda'
image_encoder = ImageEncoder(hidden_dim=ATTEN_DIM, feature_extracting=True)
image_decoder = ImageDecoder(hidden_dim=ATTEN_DIM, output_dim=IMG_SIZE)
text_encoder = TextEncoder(vocab_size=VOCABULARY_SIZE, embedding_dim=EMBED_DIM, hidden_dim=ATTEN_DIM, device=device)
text_decoder = TextDecoder(vocab_size=VOCABULARY_SIZE, embedding_dim=EMBED_DIM, hidden_dim=ATTEN_DIM, dropout=0, device=device)

att_g = AttenGround(w_map, image_encoder, text_encoder, image_decoder, text_decoder, device)
att_g.to(device)

img_loss = nn.MSELoss()
text_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(att_g.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)


def train(epoch):
    print("Training Epoch:", epoch)
    att_g.train()
    total_loss = 0
    for idx, (img, caps, info) in enumerate(train_dataset):
        img = img[None,]
        img = img.to(device)
        loss = 0
        optimizer.zero_grad()
        for s in caps:
            seq = prepare_sequence(s, w_map, device, tag=True)
            r_img, r_text, attn = att_g(img, seq)
            loss += img_loss(r_img, img) + text_loss(r_text, seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step = idx + 1
        if step % LOG_FREQ == 0:
            print("Step: %d, Loss: %.2f" % (step, total_loss/step))

for epoch in range(NUM_EPOCH):
    train(epoch)

torch.save(att_g.state_dict(), PATH)