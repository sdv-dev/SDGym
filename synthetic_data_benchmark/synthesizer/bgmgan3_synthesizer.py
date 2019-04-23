from .synthesizer_base import SynthesizerBase, run
from .synthesizer_utils import BGMTransformer, CONTINUOUS, ORDINAL, CATEGORICAL
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim
import os

class Discriminator(nn.Module):
    def __init__(self, inputDim, disDims, pack=10):
        super(Discriminator, self).__init__()
        dim = inputDim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(disDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))

class Residual(nn.Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class Generator(nn.Module):
    def __init__(self, embeddingDim, genDims, dataDim):
        super(Generator, self).__init__()
        dim = embeddingDim
        seq = []
        for item in list(genDims):
            seq += [
                Residual(dim, item)
            ]
            dim += item
        seq.append(nn.Linear(dim, dataDim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(F.tanh(data[:, st:ed]))
            # data_t.append(data[:, st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.1))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)


class Cond(object):
    def __init__(self, data, output_info):
        self.model = []

        st = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
                continue
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                assert 0
        assert st == data.shape[1]
        self.interval = []
        self.n_col = 0

        self.n_opt = 0

        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                continue
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    continue
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
            else:
                assert 0

        self.interval = np.asarray(self.interval)



    def generate(self, batch):
        if self.n_col == 0:
            return None
        batch = batch // 2
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1 = self.interval[idx, 0] + np.random.randint(1000000, size=batch) % (self.interval[idx, 1])
        vec1[np.arange(batch), opt1] = 1


        vec2 = np.zeros((batch, self.n_opt), dtype='float32')
        mask2 = np.zeros((batch, self.n_col), dtype='float32')
        mask2[np.arange(batch), idx] = 1
        opt2 = self.interval[idx, 0] + np.random.randint(1000000, size=batch) % (self.interval[idx, 1])
        vec2[np.arange(batch), opt2] = 1

        return np.concatenate([vec1, vec2], axis=0), np.concatenate([mask1, mask2], axis=0)

    def generate_zero(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec

def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    skip = False
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            skip = True
        elif item[1] == 'softmax':
            if skip:
                skip = False
                st += item [0]
                continue
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none')
            loss.append(tmp)
            st = ed
            st_c = ed_c
        else:
            assert 0
    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]

class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.weight = []

        st = 0
        w = np.zeros(len(self.data)) + 1 / len(data)
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue
                ed = st + item[0]
                w += np.sum(data[:, st:ed] / (np.sum(data[:, st:ed], axis=0) + 1e-8), axis=1) / (ed - st)
                st = ed
            else:
                assert 0
        assert st == data.shape[1]
        self.weight = w
        self.weight /= np.sum(self.weight)

    def sample(self, n):
        idx = np.random.choice(np.arange(len(self.data)), n, p=self.weight)
        return self.data[idx]



class BGMGAN3Synthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""
    def __init__(self,
                 embeddingDim=128,
                 genDim=(128, 128),
                 disDim=(128, 128),
                 l2scale=1e-6,
                 batch_size=500,
                 store_epoch=[300]):

        self.embeddingDim = embeddingDim
        self.genDim = genDim
        self.disDim = disDim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data):
        # train_data = monkey_with_train_data(train_data)
        self.transformer = BGMTransformer(self.meta)
        self.transformer.fit(train_data)
        train_data = self.transformer.transform(train_data)

        # ncp1 = sum(self.transformer.components[0])
        # ncp2 = sum(self.transformer.components[1])
        # for i in range(ncp1):
        #     for j in range(ncp2):
        #         cond1 = train_data[:, 1 + i] > 0
        #         cond2 = train_data[:, 2 + ncp1 + j]
        #         cond = np.logical_and(cond1, cond2)
        #
        #         mean1 = train_data[cond, 0].mean()
        #         mean2 = train_data[cond, 1 + ncp1].mean()
        #
        #         std1 = train_data[cond, 0].std()
        #         std2 = train_data[cond, 1 + ncp1].std()
        #         print(i, j, np.sum(cond), mean1, std1, mean2, std2, sep='\t')

        # dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        # loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        generator= Generator(self.embeddingDim + self.cond_generator.n_opt, self.genDim, data_dim).to(self.device)
        discriminator = Discriminator(data_dim, self.disDim).to(self.device)

        optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)

        max_epoch = max(self.store_epoch)
        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size//2, self.embeddingDim, device=self.device)
        std = mean + 1


        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(max_epoch):
            for id_ in range(steps_per_epoch):
                real = data_sampler.sample(self.batch_size)
                real = torch.from_numpy(real.astype('float32')).to(self.device)
                y_real = discriminator(real)

                fakez = torch.normal(mean=mean, std=std)
                fakez = torch.cat([fakez, fakez], dim=0)

                condvec = self.cond_generator.generate(self.batch_size)
                if condvec is None:
                    pass
                else:
                    c1, m1 = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)
                fake = generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)
                y_fake = discriminator(fakeact)

                loss_d = -(torch.log(torch.sigmoid(y_real) + 1e-4).mean()) - (torch.log(1. - torch.sigmoid(y_fake) + 1e-4).mean())

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)
                loss_g = -y_fake.mean() + cross_entropy

                optimizerD.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizerD.step()

                optimizerG.zero_grad()
                loss_g.backward(retain_graph=True)
                optimizerG.step()

            # print("---")
            # print(fakeact[:, 0].mean(), fakeact[:, 0].std())
            # print(fakeact[:, 1 + ncp1].mean(), fakeact[:, 1 + ncp1].std())
            print(i+1, loss_d, loss_g, cross_entropy)
            if i+1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                }, "{}/model_{}.tar".format(self.working_dir, i+1))

    def generate(self, n):
        data_dim = self.transformer.output_dim
        output_info = self.transformer.output_info
        generator= Generator(self.embeddingDim + self.cond_generator.n_opt, self.genDim, data_dim).to(self.device)

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
                fakez = torch.normal(mean=mean, std=std).to(self.device)

                condvec = self.cond_generator.generate_zero(self.batch_size)
                if condvec is None:
                    pass
                else:
                    c1 = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = generator(fakez)
                fakeact = apply_activate(fake, output_info)
                data.append(fakeact.detach().cpu().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]
            data = self.transformer.inverse_transform(data, None)
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
    run(BGMGAN3Synthesizer())
