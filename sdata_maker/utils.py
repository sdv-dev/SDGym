import seaborn as sns

def plot_samples(samples):
    sns.set(font_scale=2)
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    assert samples.shape[1] == 2
    sns.kdeplot(samples[:, 0], samples[:,1], cmap="Greens",n_levels=80, shade=True, clip=[[-5, 5]]*2)

