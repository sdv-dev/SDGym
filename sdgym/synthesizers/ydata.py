try:
    import ydata_synthetic.synthesizers.regular as ydata
except ImportError:
    ydata = None

from sdgym.synthesizers.base import SingleTableBaseline


class YData(SingleTableBaseline):

    def _fit_sample(self, real_data, table_metadata):
        if ydata is None:
            raise ImportError('Please install ydata using `make install-ydata`.')

        columns = real_data.columns
        synthesizer = self._build_ydata_synthesizer(real_data)
        synthetic_data = synthesizer.sample(len(real_data))
        synthetic_data.columns = columns

        return synthetic_data


class VanillaGAN(YData):

    def __init__(self, noise_dim=32, dim=128, batch_size=128, log_step=100,
                 epochs=201, learning_rate=5e-4, beta_1=0.5, beta_2=0.9):
        self.noise_dim = noise_dim
        self.dim = dim
        self.batch_size = batch_size
        self.log_step = log_step
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def _build_ydata_synthesizer(self, data):
        model_args = [self.batch_size, self.learning_rate, self.beta_1, self.beta_2,
                      self.noise_dim, data.shape[1], self.dim]
        train_args = ['', self.epochs, self.log_step]

        synthesizer = ydata.VanilllaGAN(model_args)
        synthesizer.train(data, train_args)

        return synthesizer


class WGAN_GP(YData):

    def __init__(self, noise_dim=32, dim=128, batch_size=128, log_step=100,
                 epochs=201, learning_rate=5e-4, beta_1=0.5, beta_2=0.9):
        self.noise_dim = noise_dim
        self.dim = dim
        self.batch_size = batch_size
        self.log_step = log_step
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def _build_ydata_synthesizer(self, data):
        model_args = [self.batch_size, self.learning_rate, self.beta_1, self.beta_2,
                      self.noise_dim, data.shape[1], self.dim]
        train_args = ['', self.epochs, self.log_step]

        synthesizer = ydata.WGAN_GP(model_args, n_critic=2)
        synthesizer.train(data, train_args)

        return synthesizer


class DRAGAN(YData):

    def __init__(self, noise_dim=128, dim=128, batch_size=500, log_step=100,
                 epochs=201, learning_rate=1e-5, beta_1=0.5, beta_2=0.9):
        self.noise_dim = noise_dim
        self.dim = dim
        self.batch_size = batch_size
        self.log_step = log_step
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def _build_ydata_synthesizer(self, data):
        gan_args = [self.batch_size, self.learning_rate, self.beta_1, self.beta_2,
                    self.noise_dim, data.shape[1], self.dim]
        train_args = ['', self.epochs, self.log_step]

        synthesizer = ydata.DRAGAN(gan_args, n_discriminator=3)
        synthesizer.train(data, train_args)

        return synthesizer


class PreprocessedDRAGAN(DRAGAN):

    CONVERT_TO_NUMERIC = True


class PreprocessedWGAN_GP(WGAN_GP):

    CONVERT_TO_NUMERIC = True


class PreprocessedVanillaGAN(VanillaGAN):

    CONVERT_TO_NUMERIC = True
