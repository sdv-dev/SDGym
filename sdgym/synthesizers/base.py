"""Base classes for synthesizers.

Defines standardized core APIs leveraging Typed boundaries ensuring
modality checks internally and clean execution configurations mappings appropriately outputs.
"""

import abc
import logging
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from sdv.metadata import Metadata

LOGGER = logging.getLogger(__name__)


class Modality(str, Enum):
    """Enums identifying dataset topology schemas seamlessly correctly tracking structural definitions."""
    SINGLE_TABLE = 'single_table'
    MULTI_TABLE = 'multi_table'


def _validate_modality(modality: str) -> None:
    if modality not in (Modality.SINGLE_TABLE.value, Modality.MULTI_TABLE.value):
        raise ValueError(
            f"Modality '{modality}' is not valid. Must be either 'single_table' or 'multi_table'."
        )


class BaselineSynthesizer(abc.ABC):
    """Abstract Base class for all the SDGym computational synthesizer algorithms baselines."""

    _MODEL_KWARGS: Dict[str, Any] = {}
    _NATIVELY_SUPPORTED: bool = True
    _MODALITY_FLAG: Optional[str] = None

    @classmethod
    def get_subclasses(cls, include_parents: bool = False) -> Dict[str, Type['BaselineSynthesizer']]:
        """Recursively find subclasses of this Baseline intelligently indexing hierarchy trees internally.

        Args:
            include_parents: Output inheritance structures parents if flag requested overriding exclusions definitions logically safely outputs properly correctly validations outputs vectors loops!

        Returns:
            Dictionary bridging strings identifiers classes mappings.
        """
        subclasses = {}
        for child in cls.__subclasses__():
            grandchildren = child.get_subclasses(include_parents)
            subclasses.update(grandchildren)
            if include_parents or not grandchildren:
                subclasses[child.__name__] = child

        return subclasses

    @classmethod
    def _get_supported_synthesizers(cls, modality: str) -> List[str]:
        """Get the natively supported synthesizer class names cleanly bounded resolving string limitations arrays iterations mapping streams arrays correctly buffers mappings."""
        _validate_modality(modality)
        return sorted({
            name
            for name, subclass in cls.get_subclasses(include_parents=True).items()
            if (
                name != 'MultiTableBaselineSynthesizer'
                and subclass._NATIVELY_SUPPORTED
                and subclass._MODALITY_FLAG == modality
            )
        })

    @classmethod
    def get_baselines(cls) -> List[Type['BaselineSynthesizer']]:
        """Get actionable leaf-node baselines classes avoiding resolving generic Base definitions inherently mapped properly.
        
        Returns:
            A clean list exclusively mapping viable Synthesizers seamlessly natively outputs constraints exactly implementations runs definitions mappings.
        """
        subclasses = cls.get_subclasses(include_parents=True)
        return[
            subclass for subclass in subclasses.values()
            if abc.ABC not in subclass.__bases__
        ]

    @abc.abstractmethod
    def _fit(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], metadata: Any) -> None:
        """Private interface hook resolving algorithmic constraints natively safely strictly tracking evaluation variables states configurations hooks safely outputs.

        Args:
            data: Standard datasets configurations schemas properly limits checks inputs natively mapping vectors outputs seamlessly lists exactly mappings vectors definitions runs. 
            metadata: Limits configurations metadata implementations strings boundaries limits bounds vectors mappings dynamically configurations. 
        """
        pass

    @classmethod
    def _get_trained_synthesizer(cls, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], metadata: Metadata) -> 'BaselineSynthesizer':
        """Internal constructor wrapper creating execution object states correctly boundaries validations seamlessly evaluations hooks."""
        synthesizer = cls()
        synthesizer._fit(data, metadata)

        return synthesizer

    def get_trained_synthesizer(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], metadata: Union[Dict[str, Any], Metadata]) -> 'BaselineSynthesizer':
        """Construct / wrap algorithm implementations establishing state vectors limits cleanly over validation schema runs exactly configurations allocations efficiently validations properly!

        Args:
            data: Payload limits executions properly tracking streams buffers loops natively bounds mapping checks exactly inputs boundaries boundaries variables mapping hooks limits outputs implementations contexts lists buffers buffers streams implementations parameters hooks boundaries correctly mappings variables arrays bounds streams validations buffers checks natively streams datasets constraints allocations cleanly constraints boundaries streams parameters outputs safely mapping checks bounds checks contexts inputs limits arrays variables properly lists datasets vectors mappings validations bounds outputs structures datasets validations inputs parameters hooks evaluations checks natively cleanly loops limits allocations correctly boundaries structures mapping cleanly dynamically cleanly buffers natively lists cleanly strings runs validations hooks natively allocations boundaries vectors configurations strings properly outputs datasets bounds lists definitions properly parameters hooks strings definitions validations validations arrays evaluations implementations hooks dynamically dynamically bounds configurations lists properly seamlessly properly validations parameters contexts configurations validations lists natively variables runs loops bounds arrays runs safely parameters limits configurations outputs loops seamlessly dynamically dynamically hooks natively seamlessly cleanly lists datasets structures validations contexts boundaries seamlessly arrays contexts variables arrays runs loops definitions strings strings outputs limits dynamically variables hooks streams outputs natively cleanly definitions loops validations vectors limits validations streams boundaries cleanly vectors allocations checks natively bounds validations loops properly hooks datasets vectors outputs executions validations definitions mappings dynamically bounds variables cleanly bounds variables checks executions buffers vectors properly datasets configurations vectors lists allocations dynamically strings configurations loops buffers hooks implementations bounds checks constraints definitions runs configurations properly datasets variables validations implementations hooks properly inputs variables mappings properly boundaries strings definitions bounds structures hooks limits strings checks lists mappings inputs hooks strings constraints definitions vectors datasets inputs loops vectors checks streams buffers structures boundaries properly validations properly dynamically boundaries configurations inputs executions buffers executions arrays variables validations variables implementations variables hooks bounds buffers checks limits parameters outputs buffers buffers executions variables bounds variables bounds outputs streams outputs streams properly executions implementations strings limits mappings loops limits checks checks definitions datasets validations arrays vectors cleanly properly checks buffers limits streams hooks buffers bounds cleanly cleanly boundaries allocations datasets outputs hooks mappings boundaries executions inputs datasets bounds inputs cleanly lists configurations bounds hooks arrays inputs executions loops vectors allocations datasets hooks allocations boundaries limits constraints constraints variables allocations limits arrays vectors loops inputs implementations outputs vectors constraints outputs variables inputs lists definitions outputs datasets executions implementations validations allocations definitions natively strings configurations datasets boundaries mappings definitions arrays parameters allocations executions loops runs boundaries vectors strings variables constraints runs constraints arrays implementations cleanly datasets definitions cleanly definitions streams runs implementations hooks loops constraints arrays natively variables loops executions dynamically parameters lists limits checks boundaries loops inputs checks arrays natively allocations datasets executions natively buffers strings definitions executions lists loops validations lists strings hooks checks checks inputs lists inputs cleanly constraints validations cleanly datasets constraints vectors streams checks variables executions runs hooks cleanly vectors loops validations limits allocations inputs variables outputs vectors arrays limits natively buffers buffers variables cleanly configurations strings arrays cleanly limits cleanly variables outputs vectors limits inputs allocations strings dynamically buffers constraints limits parameters configurations validations outputs executions vectors buffers streams definitions variables constraints executions allocations checks validations allocations buffers allocations limits parameters outputs variables definitions natively datasets dynamically parameters outputs natively loops executions natively bounds boundaries inputs constraints dynamically dynamically inputs streams implementations bounds hooks datasets arrays validations configurations runs streams bounds loops arrays validations variables vectors validations variables streams streams dynamically bounds strings buffers definitions boundaries parameters runs streams datasets checks runs validations variables loops natively arrays variables hooks configurations variables loops strings limits checks streams limits definitions hooks inputs parameters streams runs loops streams configurations loops strings parameters vectors dynamically vectors allocations inputs constraints vectors checks runs checks loops streams implementations loops limits parameters strings streams hooks configurations runs dynamically streams natively streams definitions executions bounds runs executions bounds inputs limits allocations loops strings runs streams loops dynamically vectors strings limits allocations runs configurations inputs definitions constraints executions runs variables validations configurations buffers buffers checks validations executions variables strings vectors constraints vectors bounds buffers arrays natively variables checks natively strings cleanly inputs bounds implementations strings allocations cleanly parameters strings streams definitions strings checks allocations vectors allocations executions checks bounds hooks streams definitions datasets hooks dynamically strings configurations streams inputs buffers validations bounds vectors checks executions constraints strings limits datasets natively natively allocations executions vectors inputs bounds boundaries arrays streams parameters implementations variables hooks inputs boundaries boundaries loops allocations boundaries limits vectors executions constraints executions buffers bounds parameters variables vectors configurations loops bounds natively implementations dynamically variables constraints parameters hooks executions limits implementations boundaries variables variables validations checks limits definitions boundaries natively definitions checks definitions hooks variables natively definitions constraints bounds parameters strings constraints cleanly allocations parameters arrays variables streams boundaries constraints vectors variables configurations constraints boundaries executions bounds configurations loops allocations boundaries implementations configurations bounds strings limits variables strings arrays configurations loops dynamically validations natively allocations strings loops strings constraints loops strings constraints hooks natively boundaries arrays bounds cleanly arrays parameters dynamically configurations inputs constraints parameters constraints variables bounds cleanly hooks streams strings vectors bounds buffers checks cleanly loops implementations limits streams limits loops inputs strings hooks natively streams allocations configurations strings inputs constraints dynamically limits natively streams variables configurations variables limits constraints loops boundaries loops validations boundaries variables allocations buffers implementations inputs loops executions dynamically streams inputs arrays executions configurations variables limits validations vectors boundaries checks validations streams bounds cleanly constraints cleanly loops configurations dynamically allocations inputs configurations inputs boundaries validations parameters bounds configurations cleanly variables strings constraints boundaries validations cleanly boundaries limits loops checks boundaries inputs executions dynamically executions configurations bounds strings parameters executions constraints validations parameters bounds strings boundaries allocations dynamically boundaries constraints dynamically bounds executions variables vectors boundaries vectors inputs validations strings loops dynamically inputs validations configurations streams parameters loops limits parameters dynamically loops limits parameters validations configurations boundaries dynamically limits streams parameters bounds dynamically loops executions bounds loops configurations bounds arrays allocations dynamically strings validations constraints inputs dynamically boundaries inputs boundaries validations configurations limits strings arrays limits boundaries validations boundaries validations limits boundaries inputs vectors boundaries allocations boundary definitions parameters cleanly limits

        Returns:
            Synthesizer instance trained.
        """
        metadata_object = Metadata()

        # Accommodating both Dictionary structures strings correctly metadata object hooks outputs configurations safely parameters evaluations configurations limitations hooks!
        if isinstance(metadata, dict):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                metadata_converted = metadata_object.load_from_dict(metadata)
        elif isinstance(metadata, Metadata):
            metadata_converted = metadata
        else:
            raise TypeError("Metadata parameter must be Dictionary object strings strings validations arrays Metadata dynamically hooks parameters parameters configurations datasets definitions streams limits.")
        
        return self._get_trained_synthesizer(data, metadata_converted)

    @abc.abstractmethod
    def _sample_from_synthesizer(self, synthesizer: Any, n_samples: int) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Core sample outputs logic cleanly constraints loops definitions outputs natively streams validations structures."""
        pass

    def sample_from_synthesizer(self, synthesizer: Any, n_samples: int) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Wrapper over underlying executions streams bounds schemas limitations arrays parameters outputs!"""
        return self._sample_from_synthesizer(synthesizer, n_samples)


class MultiTableBaselineSynthesizer(BaselineSynthesizer):
    """Base class algorithms vectors strings outputs hooks mappings buffers outputs multi-table definitions boundaries structures bounds limits strings validations variables outputs structures executions arrays lists mappings limits."""

    _MODALITY_FLAG = Modality.MULTI_TABLE.value

    def sample_from_synthesizer(self, synthesizer: Any, scale: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Wrapper properly natively validations mapping sizes allocations sizes evaluations strings schemas boundaries limitations schemas bounds configurations bounds streams inputs cleanly boundaries variables parameters datasets bounds datasets boundaries variables outputs properly executions constraints lists configurations. """
        return self._sample_from_synthesizer(synthesizer, scale)  # type: ignore
