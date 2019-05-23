from .synthesizer_base import SynthesizerBase, run
from .synthesizer_utils import BGMTransformer
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim
import os

class Encoder(nn.Module):
    def __init__(self, dataDim, compressDims, embeddingDim):
        super(Encoder, self).__init__()
        dim = dataDim
        seq = []
        for item in list(compressDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embeddingDim)
        self.fc2 = nn.Linear(dim, embeddingDim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar

class Decoder(nn.Module):
    def __init__(self, embeddingDim, decompressDims, dataDim):
        super(Decoder, self).__init__()
        dim = embeddingDim
        seq = []
        for item in list(decompressDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq.append(nn.Linear(dim, dataDim))
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(dataDim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma

def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            std = sigmas[st]
            loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
            loss.append(torch.log(std) * x.size()[0])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(F.cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0
    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAESynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""
    def __init__(self,
                 embeddingDim=128,
                 compressDims=(128, 128),
                 decompressDims=(128, 128),
                 l2scale=1e-5,
                 batch_size=500,
                 store_epoch=[300]):

        self.embeddingDim = embeddingDim
        self.compressDims = compressDims
        self.decompressDims = decompressDims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.store_epoch = store_epoch
        self.loss_factor = 2

    def train(self, train_data):
        self.transformer = BGMTransformer(self.meta)
        self.transformer.fit(train_data)
        train_data = self.transformer.transform(train_data)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        encoder = Encoder(data_dim, self.compressDims, self.embeddingDim).to(self.device)
        decoder = Decoder(self.embeddingDim, self.compressDims, data_dim).to(self.device)
        optimizerAE = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), weight_decay=self.l2scale)

        max_epoch = max(self.store_epoch)
        for i in range(max_epoch):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = decoder(emb)
                loss_1, loss_2 = loss_function(rec, real, sigmas, mu, logvar, self.transformer.output_info, self.loss_factor)
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                decoder.sigma.data.clamp_(0.01, 1.)
            print(i+1, loss_1, loss_2)
            if i+1 in self.store_epoch:
                torch.save({
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict()
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        data_dim = self.transformer.output_dim
        decoder = Decoder(self.embeddingDim, self.compressDims, data_dim).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            decoder.load_state_dict(checkpoint['decoder'])
            decoder.eval()
            decoder.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self.batch_size, self.embeddingDim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                fake, sigmas = decoder(noise)
                fake = torch.tanh(fake)
                data.append(fake.detach().cpu().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]
            data = self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
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


if __name__ == "__main__":
    run(TVAESynthesizer())
