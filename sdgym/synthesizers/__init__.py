from sdgym.synthesizers.clbn import CLBNSynthesizer
from sdgym.synthesizers.ctgan import CTGANSynthesizer
from sdgym.synthesizers.identity import IdentitySynthesizer
from sdgym.synthesizers.independent import IndependentSynthesizer
from sdgym.synthesizers.medgan import MedganSynthesizer
from sdgym.synthesizers.privbn import PrivBNSynthesizer
from sdgym.synthesizers.sdv import (
    CTGAN, CopulaGAN, GaussianCopulaCategorical, GaussianCopulaCategoricalFuzzy,
    GaussianCopulaOneHot)
from sdgym.synthesizers.tablegan import TableganSynthesizer
from sdgym.synthesizers.tvae import TVAESynthesizer
from sdgym.synthesizers.uniform import UniformSynthesizer
from sdgym.synthesizers.veegan import VEEGANSynthesizer

__all__ = (
    'CLBNSynthesizer',
    'IdentitySynthesizer',
    'IndependentSynthesizer',
    'MedganSynthesizer',
    'PrivBNSynthesizer',
    'TableganSynthesizer',
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'UniformSynthesizer',
    'VEEGANSynthesizer',
    'CTGAN',
    'CopulaGAN',
    'GaussianCopulaCategorical',
    'GaussianCopulaCategoricalFuzzy',
    'GaussianCopulaOneHot'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
