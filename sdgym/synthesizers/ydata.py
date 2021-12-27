import abc

from sdgym.errors import UnsupportedDataset
from sdgym.synthesizers.base import SingleTableBaseline

try:
    import ydata_synthetic as ydata
    from ydata_synthetic.synthesizers import regular
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
except ImportError:
    ydata = None


class YData(SingleTableBaseline, abc.ABC):

    SYNTHESIZER_CLASS = None
    DEFAULT_NOISE_DIM = 128
    DEFAULT_DIM = 128
    DEFAULT_BATCH_SIZE = 500
    DEFAULT_LOG_STEP = 100
    DEFAULT_EPOCHS = 301
    DEFAULT_LEARNING_RATE = 1e-5
    DEFAULT_BETA_1 = 0.5
    DEFAULT_BETA_2 = 0.9
    EXTRA_KWARGS = {}

    def __init__(self, noise_dim=None, dim=None, batch_size=None, log_step=None,
                 epochs=None, learning_rate=None, beta_1=None, beta_2=None, extra_kwargs=None):
        self.noise_dim = noise_dim or self.DEFAULT_NOISE_DIM
        self.dim = dim or self.DEFAULT_DIM
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.log_step = log_step or self.DEFAULT_LOG_STEP
        self.epochs = epochs or self.DEFAULT_EPOCHS
        self.learning_rate = learning_rate or self.DEFAULT_LEARNING_RATE
        self.beta_1 = beta_1 or self.DEFAULT_BETA_1
        self.beta_2 = beta_2 or self.DEFAULT_BETA_2
        if extra_kwargs:
            self.extra_kwargs = extra_kwargs
        elif self.EXTRA_KWARGS:
            self.extra_kwargs = self.EXTRA_KWARGS.copy()
        else:
            self.extra_kwargs = {}

    def _fit_synthesizer(self, data, numericals, categoricals):
        model_args = ModelParameters(
            batch_size=self.batch_size,
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            noise_dim=self.noise_dim,
            layers_dim=self.dim
        )
        train_args = TrainParameters(epochs=self.epochs, sample_interval=self.log_step)

        synthesizer = getattr(regular, self.SYNTHESIZER_CLASS)(model_args, **self.extra_kwargs)
        synthesizer.train(data, train_args, numericals, categoricals)

        return synthesizer

    def _fit_sample(self, real_data, table_metadata):
        if ydata is None:
            raise ImportError('Please install ydata using `make install-ydata`.')

        columns = real_data.columns
        if self.CONVERT_TO_NUMERIC:
            numericals = list(columns)
            categoricals = []
        else:
            numericals = []
            categoricals = []
            for field_name, field_meta in table_metadata['fields'].items():
                field_type = field_meta['type']
                if field_type in ('categorical', 'boolean'):
                    categoricals.append(field_name)
                elif field_type == 'numerical':
                    numericals.append(field_name)
                else:
                    raise UnsupportedDataset(f'Unsupported field type: {field_type}')

        synthesizer = self._fit_synthesizer(real_data, numericals, categoricals)
        synthetic_data = synthesizer.sample(len(real_data))
        synthetic_data.columns = columns

        return synthetic_data


class VanilllaGAN(YData):

    SYNTHESIZER_CLASS = 'VanilllaGAN'
    LEARNING_RATE = 5e-4
    EPOCHS = 201


class WGAN(YData):

    SYNTHESIZER_CLASS = 'WGAN'
    DEFAULT_LEARNING_RATE = 5e-4
    EXTRA_KWARGS = {
        'n_critic': 2
    }


class WGAN_GP(YData):

    SYNTHESIZER_CLASS = 'WGAN_GP'
    DEFAULT_LEARNING_RATE = [5e-4, 3e-3]
    EXTRA_KWARGS = {
        'n_critic': 2
    }


class DRAGAN(YData):

    SYNTHESIZER_CLASS = 'DRAGAN'
    EXTRA_KWARGS = {
        'n_discriminator': 3
    }


class CRAMERGAN(YData):

    SYNTHESIZER_CLASS = 'CRAMERGAN'
    DEFAULT_LEARNING_RATE = 5e-4
    EXTRA_KWARGS = {
        'gradient_penalty_weight': 10
    }


class PreprocessedVanilllaGAN(VanilllaGAN):

    CONVERT_TO_NUMERIC = True


class PreprocessedWGAN(WGAN):

    CONVERT_TO_NUMERIC = True


class PreprocessedWGAN_GP(WGAN_GP):

    CONVERT_TO_NUMERIC = True


class PreprocessedDRAGAN(DRAGAN):

    CONVERT_TO_NUMERIC = True


class PreprocessedCRAMERGAN(CRAMERGAN):

    CONVERT_TO_NUMERIC = True
