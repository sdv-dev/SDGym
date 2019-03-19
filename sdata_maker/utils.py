import seaborn as sns
import json
import numpy as np
CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

def plot_samples(samples):
    sns.set(font_scale=2)
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    assert samples.shape[1] == 2
    sns.kdeplot(samples[:, 0], samples[:,1], cmap="Greens",n_levels=80, shade=True, clip=[[-5, 5]]*2)

def verify_table(table, meta):
    for _id, item in enumerate(meta):
        if item['type'] == CONTINUOUS:
            assert np.all(item['min'] <= table[:, _id])
            assert np.all(table[:, _id] <= item['max'])
        else:
            assert np.all(table[:, _id].astype('int32') >= 0)
            assert np.all(table[:, _id].astype('int32') < item['size'])

def verify(datafile, metafile):
    with open(metafile) as f:
        meta = json.load(f)


    for item in meta:
        assert 'name' in item
        assert item['name'] is None or type(item['name']) == str

        assert 'type' in item
        assert item['type'] in [CATEGORICAL, CONTINUOUS, ORDINAL]

        if item['type'] == CONTINUOUS:
            assert 'min' in item and 'max' in item
        else:
            assert 'size' in item and 'i2s' in item
            assert item['size'] == len(item['i2s'])
            for ss in item['i2s']:
                assert type(ss) == str
                assert len(set(item['i2s'])) == item['size']


    data = np.load(datafile)

    verify_table(data['train'], meta)
    verify_table(data['test'], meta)
