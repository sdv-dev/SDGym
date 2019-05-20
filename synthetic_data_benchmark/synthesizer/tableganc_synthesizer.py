#!/usr/bin/env python
# coding: utf-8

# In[16]:


from .synthesizer_base import SynthesizerBase, run
from .synthesizer_utils import CATEGORICAL, ORDINAL, CONTINUOUS
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import os


class TableganTransformer(object):
    def __init__(self, meta, side):
        self.meta = meta
        self.minn = np.zeros(len(meta))
        self.maxx = np.zeros(len(meta))
        for i in range(len(meta)):
            if meta[i]['type'] == CONTINUOUS:
                self.minn[i] = meta[i]['min'] - 1e-3
                self.maxx[i] = meta[i]['max'] + 1e-3
            else:
                self.minn[i] = -1e-3
                self.maxx[i] = meta[i]['size'] - 1 + 1e-3

        self.height = side

    def fit(self, data):
        pass

    def transform(self, data):
        data = data.copy().astype('float32')
        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1
        if self.height * self.height > len(data[0]):
            padding = np.zeros((len(data), self.height * self.height - len(data[0])))
            data = np.concatenate([data, padding], axis=1)
        return data.reshape(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height)

        data_t = np.zeros([len(data), len(self.meta)])

        for id_, info in enumerate(self.meta):
            data_t[:, id_] = (data[:, id_].reshape([-1]) + 1) / 2 * (self.maxx[id_] - self.minn[id_]) + self.minn[id_]
            if info['type'] in [CATEGORICAL, ORDINAL]:
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t

# In[18]:


class Discriminator(nn.Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = nn.Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        # print("----")
        # t = input
        # print(t.shape)
        # for item in self.layers:
        #     t = item(t)
        #     print(item)
        #     print(t.shape)
        # assert 0
        return self.seq(input)


class Generator(nn.Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = nn.Sequential(*layers)
        # self.layers = layers

    def forward(self, input_):
        # print("----")
        # t = input_
        # print(t.shape)
        # for item in self.layers:
        #     t = item(t)
        #     print(item)
        #     print(t.shape)
        # assert 0
        return self.seq(input_)



class Classifier(nn.Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = nn.Sequential(*layers)
        self.valid = True
        if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32').to(device)
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        # print(side, index, self.r, self.c)
        # assert 0
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, randomDim, numChannels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (numChannels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # print(layer_dims)

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [nn.Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
                     nn.BatchNorm2d(curr[0]),
                     nn.LeakyReLU(0.2, inplace=True)]
    layers_D += [nn.Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
                 nn.Sigmoid()]

    layers_G = [nn.ConvTranspose2d(randomDim, layer_dims[-1][0], layer_dims[-1][1], 1, 0,
                                   output_padding=0, bias=False)]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [nn.BatchNorm2d(prev[0]),
                     nn.ReLU(True),
                     nn.ConvTranspose2d(prev[0], curr[0], 4, 2, 1,
                                        output_padding=0, bias=True)]
    layers_G += [nn.Tanh()]


    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [nn.Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
                     nn.BatchNorm2d(curr[0]),
                     nn.LeakyReLU(0.2, inplace=True)]
    layers_C += [nn.Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    # print(layers_D)
    # print(layers_G)

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class TableganCSynthesizer(SynthesizerBase):
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 randomDim=100,
                 numChannels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 store_epoch=[300]):

        self.randomDim = randomDim
        self.numChannels = numChannels
        self.l2scale = l2scale

        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data):
        self.transformer = TableganTransformer(self.meta, self.side)
        train_data = self.transformer.transform(train_data)
        # print(train_data[:3, 0, :, :])
        # assert 0
        train_data = torch.from_numpy(train_data.astype('float32')).to(self.device)
        dataset = torch.utils.data.TensorDataset(train_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(self.side, self.randomDim, self.numChannels)

        generator = Generator(self.meta, self.side, layers_G).to(self.device)
        discriminator = Discriminator(self.meta, self.side, layers_D).to(self.device)
        classifier = Classifier(self.meta, self.side, layers_C, self.device).to(self.device)

        optimizerG = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerC = optim.Adam(classifier.parameters(), lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)

        generator.apply(weights_init)
        discriminator.apply(weights_init)
        classifier.apply(weights_init)

        max_epoch = max(self.store_epoch)

        for i in range(max_epoch):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean())
                loss_d.backward()
                optimizerD.step()

                # print(real.size())
                # print(fake.size())

                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)
                optimizerG.zero_grad()
                y_fake = discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)
                if classifier.valid:
                    # print(real[:5, :, 3, 2])
                    real_pre, real_label = classifier(real)
                    # print(real_label)
                    # assert 0
                    fake_pre, fake_label = classifier(fake)
                    # print(real_pre, real_label)
                    loss_cc = F.binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = F.binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

                if((id_+1) % 50 == 0):
                    print("epoch", i+1, "step", id_+1, loss_d, loss_g, loss_c)
            print("epoch", i+1, "step", id_+1, loss_d, loss_g, loss_c)
            if i+1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        _, self.layers_G, _ = determine_layers(self.side, self.randomDim, self.numChannels)
        generator = Generator(self.meta, self.side, self.layers_G).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])

            generator.eval()
            generator.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)
                # print(fake.size())
                data.append(fake.detach().cpu().numpy())
            # print(data)
            data = np.concatenate(data, axis=0)
            data = self.transformer.inverse_transform(data[:n])
            # print(data.shape)
            ret.append((epoch, data))
        return ret

    def init(self, meta, working_dir):
        self.meta = meta
        self.working_dir = working_dir
        try:
            os.mkdir(working_dir)
        except:
            pass
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= len(self.meta):
                self.side = i
                break
        # figure out image size



if __name__ == "__main__":
    run(TableganCSynthesizer())
