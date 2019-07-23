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
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5)
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
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
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
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


class Cond(object):
    def __init__(self, data, output_info):
        # self.n_col = self.n_opt = 0
        # return
        self.model = []

        st = 0
        skip = False
        max_interval = 0
        counter = 0
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
                max_interval = max(max_interval, ed - st)
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                assert 0
        assert st == data.shape[1]


        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        st = 0
        self.p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                st += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    st += item[0]
                    skip = False
                    continue
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
            else:
                assert 0
        self.interval = np.asarray(self.interval)


    def generate(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

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
        self.model = []
        self.n = len(data)

        st = 0
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
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
            else:
                assert 0
        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]

LAMBDA = 10

def calc_gradient_penalty(netD, real_data, fake_data, device='cpu', pac=10):
    alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
    alpha = alpha.repeat(1, pac, real_data.size(1))
    alpha = alpha.view(-1, real_data.size(1))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # interpolates = torch.Variable(interpolates, requires_grad=True, device=device)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



class TGANSynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""
    def __init__(self,
                 embeddingDim=128,
                 genDim=(256, 256),
                 disDim=(256, 256),
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
        discriminator = Discriminator(data_dim + self.cond_generator.n_opt, self.disDim).to(self.device)

        optimizerG = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))#, weight_decay=self.l2scale)

        max_epoch = max(self.store_epoch)
        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embeddingDim, device=self.device)
        std = mean + 1


        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(max_epoch):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.generate(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)


                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake


                # print(real_cat[0])
                # print(fake_cat[0])
                # assert 0

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)


                # loss_d = -(torch.log(torch.sigmoid(y_real) + 1e-4).mean()) - (torch.log(1. - torch.sigmoid(y_fake) + 1e-4).mean())
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                # for p in discriminator.parameters():
                    # p.data.clamp_(-0.05, 0.05)

                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.generate(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)
                fake = generator(fakez)

                fakeact = apply_activate(fake, self.transformer.output_info)
                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)
                # loss_g = -torch.log(torch.sigmoid(y_fake) + 1e-4).mean() + cross_entropy
                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()


            # print("---")
            # print(fakeact[:, 0].mean(), fakeact[:, 0].std())
            # print(fakeact[:, 1 + ncp1].mean(), fakeact[:, 1 + ncp1].std())
            print(i+1, loss_d.data, pen.data, loss_g.data, cross_entropy)
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
    run(TGANSynthesizer())
