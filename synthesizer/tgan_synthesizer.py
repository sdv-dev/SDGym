from synthesizer_base import SynthesizerBase, run
from utils import CATEGORICAL, ORDINAL, CONTINUOUS
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import os

class TganTransformer(object):
    def __init__(self, meta, n_clusters=5):
        self.meta = meta
        self.n_clusters = n_clusters

    def fit(self, data):
        model = []

        self.output_info = []
        self.output_stack_dim = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                gm = GaussianMixture(self.n_clusters)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                self.output_info += [(1, 'tanh'), (self.n_clusters, 'softmax')]
                self.output_stack_dim += 1 + self.n_clusters
            else:
                model.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_stack_dim += info['size']

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == CONTINUOUS:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (2 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                argmax = np.argmax(probs, axis=1)
                idx = np.arange((len(features)))
                features = features[idx, argmax].reshape([-1, 1])

                features = np.clip(features, -.99, .99)

                values += [features, probs]
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), current.astype('int32')] = 1
                values.append(col_t)

        return values

    def inverse_transform(self, data):
        data_t = np.zeros([len(data[0]), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                u = data[st].reshape([-1])
                v = data[st + 1]
                st += 2
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 2 * std_t  + mean_t
                data_t[:, id_] = tmp
            else:
                current = data[st]
                st += 1

                data_t[:, id_] = np.argmax(current, axis=1)
        return data_t

def get_activate(x):
    if x == 'tanh':
        return nn.Tanh()
    if x == 'relu':
        return nn.ReLU()
    if x == 'softmax':
        return nn.Softmax(-1)
    assert 0

class Generator(nn.Module):
    def __init__(self, randomDim, hiddenDim, outputInfo):
        super(Generator, self).__init__()
        self.outputInfo = outputInfo

        self.rnn = nn.LSTMCell(randomDim, hiddenDim)
        self.fcs = nn.ModuleList()
        for info in outputInfo:
            fc = nn.Sequential(
                        nn.Linear(hiddenDim, hiddenDim),
                        nn.ReLU(),
                        nn.Linear(hiddenDim, info[0]),
                        get_activate(info[1]))

            self.fcs.append(fc)


    def forward(self, input):
        states = None

        outputs = []
        for fc in self.fcs:
            states = self.rnn(input)
            output = fc(states[0])
            outputs.append(output)

        return outputs

class Discriminator(nn.Module):
    def __init__(self, dataDim, hiddenDim):
        super(Discriminator, self).__init__()
        dim = dataDim
        seq = []
        for item in list(hiddenDim):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU() if item > 1 else nn.Sigmoid()
            ]
            dim = item
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        input_concat = torch.cat(input, dim=1)
        return self.seq(input_concat)


class TganSynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""
    def __init__(self,
                 randomDim=128,
                 generatorDim=256,                 # RNN 200, FC 200
                 discriminatorDims=(256, 256, 1),   # datadim -> 256 -> 256 -> 1
                 l2scale=1e-5,
                 batch_size=1000,
                 store_epoch=[50, 100, 200]):

        self.randomDim = randomDim
        self.generatorDim = generatorDim
        self.discriminatorDims = discriminatorDims
        self.l2scale = l2scale

        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data):
        self.transformer = TganTransformer(self.meta)
        self.transformer.fit(train_data)
        train_data = self.transformer.transform(train_data)
        train_data = [torch.from_numpy(item.astype('float32')).to(self.device) for item in train_data]
        dataset = torch.utils.data.TensorDataset(*train_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        n_batch = len(train_data) // self.batch_size

        output_info = self.transformer.output_info
        data_dim = self.transformer.output_stack_dim

        generator = Generator(self.randomDim, self.generatorDim, output_info).to(self.device)
        discriminator = Discriminator(data_dim, self.discriminatorDims).to(self.device)
        optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)

        max_epoch = max(self.store_epoch)
        mean = torch.zeros(self.batch_size, self.randomDim, device=self.device)
        std = mean + 1

        for i in range(max_epoch):
            n_d = 2
            n_g = 1
            for id_, data in enumerate(loader):
                real = [item.to(self.device) for item in data]
                noise = torch.normal(mean=mean, std=std)
                fake = generator(noise)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = -(torch.log(y_real + 1e-12).mean()) - (torch.log(1. - y_fake + 1e-12).mean())
                loss_d.backward()
                optimizerD.step()

                if (id_ + 1) % n_d == 0:
                    for _ in range(n_g):
                        noise = torch.normal(mean=mean, std=std)
                        fake = generator(noise)
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
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        output_info = self.transformer.output_info
        generator = Generator(self.randomDim, self.generatorDim, output_info).to(self.device)

        ret = []
        mean = torch.zeros(self.batch_size, self.randomDim, device=self.device)
        std = mean + 1
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])

            generator.eval()
            generator.to(self.device)

            steps = n // self.batch_size + 1
            data = [[] for i in range(len(output_info))]
            for i in range(steps):
                noise = torch.normal(mean=mean, std=std)
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


if __name__ == "__main__":
    run(TganSynthesizer())
