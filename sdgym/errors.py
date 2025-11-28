"""Errors for SDGym."""


class SDGymError(Exception):
    """Known error that is contemplated in the SDGym workflows."""


class UnsupportedDataset(SDGymError):
    """The Dataset is not supported by this Synthesizer."""


class BenchmarkError(RuntimeError):
    """Error raised during benchmarking."""

    def __init__(
        self,
        original_exc,
        train_time,
        sample_time,
        synthesizer_size,
        peak_memory,
        exception_text,
        error_text,
    ):
        super().__init__(str(original_exc))
        self.original_exc = original_exc
        self.train_time = train_time
        self.sample_time = sample_time
        self.synthesizer_size = synthesizer_size
        self.peak_memory = peak_memory
        self.exception = exception_text
        self.error = error_text
