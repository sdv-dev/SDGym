from .synthesizer_base import SynthesizerBase, run
from .synthesizer_utils import GeneralTransformer
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim
import os


class Reconstructor(nn.Module):

    def __init__(self, dataDim, recDims, embeddingDim):
        super(Reconstructor, self).__init__()
        dim = dataDim
        seq = []
        for item in list(recDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq += [nn.Linear(dim, embeddingDim)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(nn.Module):

    def __init__(self, inputDim, disDims):
        super(Discriminator, self).__init__()
        dim = inputDim
        seq = []
        for item in list(disDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU(),
                nn.Dropout(0.5)
            ]
            dim = item
        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Generator(nn.Module):

    def __init__(self, embeddingDim, genDims, dataDim):
        super(Generator, self).__init__()
        dim = embeddingDim
        seq = []
        for item in list(genDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq.append(nn.Linear(dim, dataDim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input, output_info):
        data = self.seq(input)
        data_t = []
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(F.softmax(data[:, st:ed], dim=1))
                st = ed
            else:
                assert 0
        return torch.cat(data_t, dim=1)


class VEEGANSynthesizer(SynthesizerBase):
    """docstring for VEEGANSynthesizer."""

    def __init__(self,
                 embeddingDim=32,
                 genDim=(128, 128),
                 disDim=(128, ),
                 recDim=(128, 128),
                 l2scale=1e-6,
                 batch_size=500,
                 store_epoch=[300]):

        self.embeddingDim = embeddingDim
        self.genDim = genDim
        self.disDim = disDim
        self.recDim = recDim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data):
        self.transformer = GeneralTransformer(self.meta, act='tanh')
        self.transformer.fit(train_data)
        train_data = self.transformer.transform(train_data)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        generator = Generator(self.embeddingDim, self.genDim, data_dim).to(self.device)
        discriminator = Discriminator(self.embeddingDim + data_dim, self.disDim).to(self.device)
        reconstructor = Reconstructor(data_dim, self.recDim, self.embeddingDim).to(self.device)

        optimizerG = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerR = optim.Adam(reconstructor.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=self.l2scale)

        max_epoch = max(self.store_epoch)
        mean = torch.zeros(self.batch_size, self.embeddingDim, device=self.device)
        std = mean + 1
        for i in range(max_epoch):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                realz = reconstructor(real)
                y_real = discriminator(torch.cat([real, realz], dim=1))

                fakez = torch.normal(mean=mean, std=std)
                fake = generator(fakez, self.transformer.output_info)
                fakezrec = reconstructor(fake)
                y_fake = discriminator(torch.cat([fake, fakez], dim=1))

                loss_d = -(torch.log(torch.sigmoid(y_real) + 1e-4).mean()) - (torch.log(1. - torch.sigmoid(y_fake) + 1e-4).mean())
                loss_g = -y_fake.mean() + F.mse_loss(fakezrec, fakez, reduction='mean') / self.embeddingDim
                loss_r = -y_fake.mean() + F.mse_loss(fakezrec, fakez, reduction='mean') / self.embeddingDim
                optimizerD.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerG.zero_grad()
                loss_g.backward(retain_graph=True)
                optimizerG.step()
                optimizerR.zero_grad()
                loss_r.backward()
                optimizerR.step()
            print(i, loss_d, loss_g, loss_r)
            if i+1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "reconstructor": reconstructor.state_dict(),
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        data_dim = self.transformer.output_dim
        output_info = self.transformer.output_info
        generator = Generator(self.embeddingDim, self.genDim, data_dim).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])
            generator.eval()
            generator.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self.batch_size, self.embeddingDim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                fake = generator(noise, output_info)
                data.append(fake.detach().cpu().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]
            data = self.transformer.inverse_transform(data)
            ret.append((epoch, data))
        return ret

    def init(self, meta, working_dir):
        self.meta = meta
        self.working_dir = working_dir

        self.embeddingDim = min(self.embeddingDim, len(self.meta))
        try:
            os.mkdir(working_dir)
        except FileExistsError:
            pass

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    run(VEEGANSynthesizer())
