import os.path
import random
import pyarrow as pa
import numpy as np
import torch
import torchvision
from PIL import Image
from hydra.utils import to_absolute_path, get_original_cwd
import lmdb
import ast
import io


class TextLineDataset(torch.utils.data.Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.root = root
        # self._env = None
        self._nSamples = 0        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        
    
        if not self.env:
            print(f'cannot creat lmdb from {self.root}')
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples
        return self.nSamples
        
    @property
    def nSamples(self):
        return self._nSamples
    
    @nSamples.setter
    def nSamples(self,value):
        self._nSamples = value

    
    def get_env(self):
        return lmdb.open(
            self.root,
            max_readers=5,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
    

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        idx = index+1 # note that the lmdb indexing starts from 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % idx
            label_key = 'label-%09d' % idx

            imgbuf = txn.get(img_key.encode('utf-8'))
            imgbuf = ast.literal_eval(imgbuf.decode())
            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except :
                print('Corrupted image for %d' % index)
                return self[index + 1]
        #     label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8')).decode()
     

        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            label = self.target_transform(label)
        
        #ORD = lambda x : ''.join([str(ord(c)) for c in x])
        #label = ORD(label)
        return img, label
    @property
    def env(self):
        
        return self.get_env()
    


class ResizeNormalize(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        if len(img.size) == 2:
            c = 1
        else:
            c = 3

        h, w = img.height, img.width
        height = self.img_height
        width = int(w * height / h)
        if width >= self.img_width:
            img = img.resize((self.img_width, self.img_height))
            img = np.array(img)
        else:
            img = img.resize((width, height))
            img = np.array(img)
            if c == 1:
                img_pad = np.zeros((self.img_height, self.img_width), dtype=img.dtype)
                img_pad[:height, :width] = img
            else:
                img_pad = np.zeros((self.img_height, self.img_width, c), dtype=img.dtype)
                img_pad[:height, :width, :] = img
            img = img_pad

        img = Image.fromarray(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batches = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batches):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail

        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[n_batches * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):

    def __init__(self, img_height=32, img_width=512):
        self.img_height = img_height
        self.img_width = img_width
        self.transform = ResizeNormalize(img_width=self.img_width, img_height=self.img_height)

    def __call__(self, batch):
        images, labels = zip(*batch)

        images = [self.transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
