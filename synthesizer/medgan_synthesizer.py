from synthesizer_base import SynthesizerBase, run
from utils import GeneralTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import os

class ResidualFC(nn.Module):
    def __init__(self, inputDim, outputDim, activate, bnDecay):
        super(ResidualFC, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(inputDim, outputDim),
            nn.BatchNorm1d(outputDim, momentum=bnDecay),
            activate()
        )

    def forward(self, input):
        residual = self.seq(input)
        return input + residual

class Generator(nn.Module):
    def __init__(self, randomDim, hiddenDim, bnDecay):
        super(Generator, self).__init__()

        dim = randomDim
        seq = []
        for item in list(hiddenDim)[:-1]:
            assert item == dim
            seq += [ResidualFC(dim, dim, nn.ReLU, bnDecay)]
        assert hiddenDim[-1] == dim
        seq += [
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, momentum=bnDecay),
            nn.ReLU()
        ]
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        return self.seq(input)

class Discriminator(nn.Module):
    def __init__(self, dataDim, hiddenDim):
        super(Discriminator, self).__init__()
        dim = dataDim * 2
        seq = []
        for item in list(hiddenDim):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU() if item > 1 else nn.Sigmoid()
            ]
            dim = item
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        mean = input.mean(dim=0, keepdim=True)
        mean = mean.expand_as(input)
        inp = torch.cat((input, mean), dim=1)
        return self.seq(inp)


class Encoder(nn.Module):
    def __init__(self, dataDim, compressDims, embeddingDim):
        super(Encoder, self).__init__()
        dim = dataDim
        seq = []
        for item in list(compressDims) + [embeddingDim]:
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        return self.seq(input)

class Decoder(nn.Module):
    def __init__(self, embeddingDim, decompressDims, dataDim):
        super(Decoder, self).__init__()
        dim = embeddingDim
        seq = []
        for item in list(decompressDims) + [dataDim]:
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class MedganSynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""
    def __init__(self,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),          # 128 -> 128 -> 128
                 discriminatorDims=(256, 128, 1),   # datadim * 2 -> 256 -> 128 -> 1
                 compressDims=(),                   # datadim -> embeddingDim
                 decompressDims=(),                 # embeddingDim -> datadim
                 bnDecay=0.99,
                 l2scale=0.001,
                 pretrain_epoch=100,
                 batch_size=1000,
                 store_epoch=[50, 100, 200]):

        self.embeddingDim = embeddingDim
        self.randomDim = randomDim
        self.generatorDims = generatorDims
        self.discriminatorDims = discriminatorDims

        self.compressDims = compressDims
        self.decompressDims = decompressDims
        self.bnDecay = bnDecay
        self.l2scale = l2scale

        self.pretrain_epoch = pretrain_epoch
        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data):
        self.transformer = GeneralTransformer(self.meta)
        self.transformer.fit(train_data)
        train_data = self.transformer.transform(train_data)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data.astype('float32')))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)


        n_batch = len(train_data) // self.batch_size

        data_dim = self.transformer.output_dim
        # print(train_data.shape, data_dim)
        # assert 0
        encoder = Encoder(data_dim, self.compressDims, self.embeddingDim).to(self.device)
        decoder = Decoder(self.embeddingDim, self.compressDims, data_dim).to(self.device)
        mseloss = nn.MSELoss().to(self.device)
        optimizerAE = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), weight_decay=self.l2scale)


        for i in range(self.pretrain_epoch):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                emb = encoder(real)
                rec = decoder(emb)
                loss = mseloss(rec, real)
                loss.backward()
                optimizerAE.step()
            print(loss)

        generator = Generator(self.randomDim, self.generatorDims, self.bnDecay).to(self.device)
        discriminator = Discriminator(data_dim, self.discriminatorDims).to(self.device)
        optimizerG = optim.Adam(generator.parameters(), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), weight_decay=self.l2scale)

        max_epoch = max(self.store_epoch)
        for i in range(max_epoch):
            n_d = 2
            n_g = 1
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                mean = torch.zeros(self.batch_size, self.randomDim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                emb = generator(noise)
                fake = decoder(emb)

                if id_ % (n_d + n_g) < n_d:
                    optimizerD.zero_grad()
                    y_real = discriminator(real)
                    y_fake = discriminator(fake)
                    loss_d = -(torch.log(y_real + 1e-12).mean()) - (torch.log(1. - y_fake + 1e-12).mean())
                    loss_d.backward()
                    optimizerD.step()
                else:
                    optimizerG.zero_grad()
                    y_fake = discriminator(fake)
                    loss_g = -(torch.log(y_fake + 1e-12).mean())
                    loss_g.backward()
                    optimizerG.step()

            print(loss_d, loss_g)
            if i+1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict()
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        data_dim = self.transformer.output_dim
        generator = Generator(self.randomDim, self.generatorDims, self.bnDecay).to(self.device)
        decoder = Decoder(self.embeddingDim, self.compressDims, data_dim).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])
            decoder.load_state_dict(checkpoint['decoder'])

            generator.eval()
            decoder.eval()

            generator.to(self.device)
            decoder.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self.batch_size, self.randomDim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                emb = generator(noise)
                fake = decoder(emb)
                data.append(fake.detach().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]
            data = self.transformer.inverse_transform(data)
            ret.append((epoch, data))
        return ret

    def init(self, meta, working_dir):
        self.meta = meta
        self.working_dir = working_dir
        os.mkdir(working_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    run(MedganSynthesizer())
