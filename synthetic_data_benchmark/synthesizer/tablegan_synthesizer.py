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


# In[17]:


class TableganTransformer(object):
    def __init__(self, meta):
        self.meta = meta

    def transform(self, data):
        values = []
        count = 0
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == CONTINUOUS:
                values.append(current.reshape([-1, 1]))
                count += 1
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), current.astype('int32')] = 1
                values.append(col_t)
                count += info['size']

        self.variable_count = count
        return values

    def inverse_transform(self, data):
        data_t = np.zeros([len(data[0]), len(self.meta)])

        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                data_t[:, id_] = data[id_].reshape([-1])
            else:
                data_t[:, id_] = np.argmax(data[id_], axis=1)

        return data_t


# In[18]:


class Discriminator(nn.Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = nn.Sequential(*layers)

    def forward(self, input_):
        input_concat = torch.cat(input_, dim=1)
        input_padded = F.pad(input_concat, (0, self.side * self.side - input_concat.shape[1]))
        input_square = input_padded.reshape((input_padded.shape[0], 1, self.side, self.side))
        return self.seq(input_square)


# In[19]:


class Generator(nn.Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = nn.Sequential(*layers)

    def forward(self, input_):
        output_square = self.seq(input_)
        output_padded = output_square.reshape((output_square.shape[0], -1))

        output = []
        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                output.append(output_padded[:, st:st+1])
                st += 1
            else:
                col_t = output_padded[:, st:st+info['size']];
                output.append(nn.Softmax(dim=1)(col_t))
                st += info['size']
        return output


# In[20]:


def determine_layers(side, randomDim, numChannels):
    assert side >= 4 and side <= 31

    layer_dims = [(1, side), (numChannels, side // 2)]
    while layer_dims[-1][1] > 3:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [nn.Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
                     nn.BatchNorm2d(curr[0]),
                     nn.LeakyReLU(0.2, inplace=True)]
    layers_D += [nn.Conv2d(layer_dims[-1][0], 1, 4, 2, 1, bias=False),
                 nn.Sigmoid()]

    layers_G = [nn.ConvTranspose2d(randomDim, layer_dims[-1][0], 4, 2, 1,
                                   output_padding=(layer_dims[-1][1] - 2), bias=False)]
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [nn.BatchNorm2d(prev[0]),
                     nn.ReLU(True),
                     nn.ConvTranspose2d(prev[0], curr[0], 4, 2, 1,
                                        output_padding=(curr[1] - prev[1]*2), bias=False)]

    return layers_D, layers_G


# In[21]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[22]:


class TableganSynthesizer(SynthesizerBase):
    """docstring for TableganSynthesizer??"""

    supported_datasets = ['credit']

    def __init__(self,
                 randomDim=100,
                 numChannels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 store_epoch=[10, 20, 50]):

        self.randomDim = randomDim
        self.numChannels = numChannels
        self.l2scale = l2scale

        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data):
        self.transformer = TableganTransformer(self.meta)
        train_data = self.transformer.transform(train_data)
        train_data = [torch.from_numpy(item.astype('float32')).to(self.device) for item in train_data]
        dataset = torch.utils.data.TensorDataset(*train_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        variable_count = self.transformer.variable_count
        self.side = int((variable_count - 1) ** 0.5) + 1  # ceil(sqrt(variable_count))

        layers_D, layers_G = determine_layers(self.side, self.randomDim, self.numChannels)

        generator = Generator(self.meta, self.side, layers_G).to(self.device)
        discriminator = Discriminator(self.meta, self.side, layers_D).to(self.device)
        optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        generator.apply(weights_init)
        discriminator.apply(weights_init)

        max_epoch = max(self.store_epoch)

        for i in range(max_epoch):
            for id_, data in enumerate(loader):
                real = [item.to(self.device) for item in data]
                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = -(torch.log(y_real + 1e-12).mean()) - (torch.log(1. - y_fake + 1e-12).mean())
                loss_d.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)
                optimizerG.zero_grad()
                y_fake = discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-12).mean())
                loss_g.backward()
                optimizerG.step()

                if((id_+1) % 50 == 0):
                    print(i+1, id_+1, loss_d, loss_g)

            print(i+1, loss_d, loss_g)
            if i+1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        _, self.layers_G = determine_layers(self.side, self.randomDim, self.numChannels)
        generator = Generator(self.meta, self.side, self.layers_G).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])

            generator.eval()
            generator.to(self.device)

            steps = n // self.batch_size + 1
            data = [[] for i in range(len(self.meta))]
            for i in range(steps):
                noise = torch.randn(self.batch_size, self.randomDim, 1, 1, device=self.device)
                fake = generator(noise)
                for id_, item in enumerate(fake):
                    data[id_].append(item.detach().cpu().numpy())

            data = [np.concatenate(item, axis=0)[:n] for item in data]
            data = self.transformer.inverse_transform(data)
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


# In[23]:


if __name__ == "__main__":
    run(TableganSynthesizer())


# In[25]:


#get_ipython().system('jupyter nbconvert --to script tablegan_synthesizer.ipynb')


# In[ ]:





# In[ ]:
