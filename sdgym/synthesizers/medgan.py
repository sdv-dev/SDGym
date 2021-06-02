import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Module, Sequential
from torch.nn.functional import cross_entropy, mse_loss, sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sdgym.synthesizers.base import LegacySingleTableBaseline
from sdgym.synthesizers.utils import GeneralTransformer, select_device


class ResidualFC(Module):
    def __init__(self, input_dim, output_dim, activate, bn_decay):
        super(ResidualFC, self).__init__()
        self.seq = Sequential(
            Linear(input_dim, output_dim),
            BatchNorm1d(output_dim, momentum=bn_decay),
            activate()
        )

    def forward(self, input):
        residual = self.seq(input)
        return input + residual


class Generator(Module):
    def __init__(self, random_dim, hidden_dim, bn_decay):
        super(Generator, self).__init__()

        dim = random_dim
        seq = []
        for item in list(hidden_dim)[:-1]:
            assert item == dim
            seq += [ResidualFC(dim, dim, nn.ReLU, bn_decay)]
        assert hidden_dim[-1] == dim
        seq += [
            Linear(dim, dim),
            BatchNorm1d(dim, momentum=bn_decay),
            nn.ReLU()
        ]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(Module):
    def __init__(self, data_dim, hidden_dim):
        super(Discriminator, self).__init__()
        dim = data_dim * 2
        seq = []
        for item in list(hidden_dim):
            seq += [
                Linear(dim, item),
                nn.ReLU() if item > 1 else nn.Sigmoid()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        mean = input.mean(dim=0, keepdim=True)
        mean = mean.expand_as(input)
        inp = torch.cat((input, mean), dim=1)
        return self.seq(inp)


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims) + [embedding_dim]:
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input, output_info):
        return self.seq(input)


def aeloss(fake, real, output_info):
    st = 0
    loss = []
    for item in output_info:
        if item[1] == 'sigmoid':
            ed = st + item[0]
            loss.append(mse_loss(sigmoid(fake[:, st:ed]), real[:, st:ed], reduction='sum'))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                fake[:, st:ed], torch.argmax(real[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0
    return sum(loss) / fake.size()[0]


class MedGAN(LegacySingleTableBaseline):
    """docstring for MedGAN."""

    def __init__(self,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),          # 128 -> 128 -> 128
                 discriminator_dims=(256, 128, 1),   # datadim * 2 -> 256 -> 128 -> 1
                 compress_dims=(),                   # datadim -> embedding_dim
                 decompress_dims=(),                 # embedding_dim -> datadim
                 bn_decay=0.99,
                 l2scale=0.001,
                 pretrain_epoch=200,
                 batch_size=1000,
                 epochs=2000):

        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims

        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.bn_decay = bn_decay
        self.l2scale = l2scale

        self.pretrain_epoch = pretrain_epoch
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = select_device()
        self.transformer = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.transformer = GeneralTransformer()
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        data = self.transformer.transform(data)
        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )

        for i in range(self.pretrain_epoch):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                emb = encoder(real)
                rec = self.decoder(emb, self.transformer.output_info)
                loss = aeloss(rec, real, self.transformer.output_info)
                loss.backward()
                optimizerAE.step()

        self.generator = Generator(
            self.random_dim, self.generator_dims, self.bn_decay).to(self.device)
        discriminator = Discriminator(data_dim, self.discriminator_dims).to(self.device)
        optimizerG = Adam(
            list(self.generator.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )
        optimizerD = Adam(discriminator.parameters(), weight_decay=self.l2scale)

        mean = torch.zeros(self.batch_size, self.random_dim, device=self.device)
        std = mean + 1
        for i in range(self.epochs):
            n_d = 2
            n_g = 1
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.normal(mean=mean, std=std)
                emb = self.generator(noise)
                fake = self.decoder(emb, self.transformer.output_info)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                real_loss = -(torch.log(y_real + 1e-4).mean())
                fake_loss = (torch.log(1.0 - y_fake + 1e-4).mean())
                loss_d = real_loss - fake_loss
                loss_d.backward()
                optimizerD.step()

                if i % n_d == 0:
                    for _ in range(n_g):
                        noise = torch.normal(mean=mean, std=std)
                        emb = self.generator(noise)
                        fake = self.decoder(emb, self.transformer.output_info)
                        optimizerG.zero_grad()
                        y_fake = discriminator(fake)
                        loss_g = -(torch.log(y_fake + 1e-4).mean())
                        loss_g.backward()
                        optimizerG.step()

    def sample(self, n):
        self.generator.eval()
        self.decoder.eval()

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.random_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            emb = self.generator(noise)
            fake = self.decoder(emb, self.transformer.output_info)
            fake = torch.sigmoid(fake)
            data.append(fake.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data)
