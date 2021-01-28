"""Errors for SDGym."""


class SDGymError(Exception):
    """Known error that is contemplated in the SDGym workflows."""


class SDGymTimeout(SDGymError):
    """The process took too long and reached a timeout."""


class UnsupportedDataset(SDGymError):
    """The Dataset is not supported by this Synthesizer."""
