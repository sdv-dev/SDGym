"""Errors for SDGym."""


class SDGymError(Exception):
    """Known error that is contemplated in the SDGym workflows."""


class UnsupportedDataset(SDGymError):
    """The Dataset is not supported by this Synthesizer."""


class SynthesisRunError(RuntimeError):
    """Error to raise when there is an error during the benchmark."""

    def __init__(
        self,
        *,
        original_exc,
        synthetic_data,
        train_time,
        sample_time,
        synthesizer_size,
        peak_memory,
        exception_text,
        error_text,
    ):
        super().__init__(str(original_exc))
        self.original_exc = original_exc
        self.synthetic_data = synthetic_data
        self.train_time = train_time
        self.sample_time = sample_time
        self.synthesizer_size = synthesizer_size
        self.peak_memory = peak_memory
        self.exception = exception_text
        self.error = error_text
