from collections import defaultdict
from PIL import Image
import glob
import os

import torch
import torch.utils.data as data

from const import START_TAG, END_TAG

class Flickr30k(data.Dataset):

    def __init__(self, root, ann_root, train, vocab_size, w_map=None, transform=None):
        super(Flickr30k, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.ann_root = os.path.expanduser(ann_root)
    
        # Ids
        self.ids = list()
        if train:
            ids_file = os.path.join(self.ann_root, 'train.txt')
        else:
            ids_file = os.path.join(self.ann_root, 'test.txt')
        with open(ids_file) as f:
            for line in f:
                self.ids.append(line.strip()+'.jpg')
        self.ids = sorted(self.ids)
        ids_set = set(self.ids)
        
        # Sentences
        sentence_file = os.path.join(self.ann_root, 'results_20130124.token')
        self.annotations = defaultdict(list)
        self.w_count = defaultdict(int)
        with open(sentence_file) as f:
            for line in f:
                img_id, caption = line.strip().split('\t')
                img_id = img_id[:-2]
                if img_id in ids_set:
                    self.annotations[img_id].append(caption)
                    ws = caption.lower().split(' ')
                    for w in ws:
                        self.w_count[w] += 1
        if w_map is None:
            top_w = sorted(list(self.w_count.items()), key=lambda t: t[1], reverse=True)[:vocab_size-2]
            self.w_map = {START_TAG:0, END_TAG:1}
            for w, _ in top_w:
                self.w_map[w] = len(self.w_map)
        else:
            self.w_map = w_map
    
    def get_w_map(self):
        return self.w_map

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert('RGB')
        width, height = img.size
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        sentences = self.annotations[img_id]
        s_tokens = []
        for sentence in sentences:
            tokens = [w for w in sentence.lower().split() if w in self.w_map]
            s_tokens.append(tokens)
        
        img_info = {'width': width, 'height': height, 'img_id': img_id, 'sentences': sentences}

        return img, s_tokens, img_info

    def __len__(self):
        return len(self.ids)


class Flickr30k_stats():

    def __init__(self, root, ann_root):
        super(Flickr30k_stats, self).__init__()
        self.root = os.path.expanduser(root)
        self.ann_root = os.path.expanduser(ann_root)
 
        # Sentences
        sentence_file = os.path.join(self.ann_root, 'results_20130124.token')
        self.annotations = defaultdict(list)
        self.w_map = {'<END>':0, '<PAD>': 1}
        self.w_count = defaultdict(int)
        max_l_sentence = 0
        sentence = []
        with open(sentence_file) as f:
            for line in f:
                img_id, caption = line.strip().split('\t')
                img_id = img_id[:-2]
                self.annotations[img_id].append(caption)
                ws = caption.split(' ')
                max_l_sentence = max(max_l_sentence, len(ws))
                sentence.append([len(ws), caption])
                for w in ws:
                    self.w_count[w] += 1
                    if w not in self.w_map:
                        self.w_map[w] = len(self.w_map)
        self.max_len = max_l_sentence
        self.sentence = sorted(sentence, reverse=True)