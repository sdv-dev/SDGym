import logging

from sdgym.benchmark import benchmark
from sdgym.synthesizers import (
    CLBNSynthesizer, CTGANSynthesizer, IdentitySynthesizer, IndependentSynthesizer,
    MedganSynthesizer, PrivBNSynthesizer, TableganSynthesizer, TVAESynthesizer, UniformSynthesizer,
    VEEGANSynthesizer)

EPOCHS_SYNTHS = (
    CTGANSynthesizer,
    MedganSynthesizer,
    TableganSynthesizer,
    TVAESynthesizer,
    VEEGANSynthesizer,
)


NO_INIT = (
    CLBNSynthesizer,
    IndependentSynthesizer,
    IdentitySynthesizer,
    UniformSynthesizer,
    PrivBNSynthesizer,
)


if __name__ == '__main__':
    # This is to be run locally by hand, as some synthesizers take
    # a long time and might fail in travis
    #
    # Run as:
    #
    #     $ python tests/test_synthesizers.py

    logging.basicConfig(level=logging.INFO)

    for synthesizer_class in EPOCHS_SYNTHS:
        synthesizer = synthesizer_class(epochs=1)
        benchmark(synthesizer.fit_sample, datasets=['adult'], repeat=1)

    for synthesizer_class in NO_INIT:
        synthesizer = synthesizer_class()
        benchmark(synthesizer.fit_sample, datasets=['adult'], repeat=1)

    logging.info('All the synthesizers were executed successfully')
