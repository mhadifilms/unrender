"""Optional audio-analysis provider adapters.

Importing this package is side-effect free: model runtimes are imported and
weights are resolved only when an adapter method is executed.
"""

from .beat_this import CHORD_VOCABULARY, BeatThisProvider
from .brouhaha import BrouhahaProvider
from .clap import CLAPProvider, ClapProvider
from .contract import (
    AudioAnalysisProvider,
    CapabilityState,
    CapabilityStatus,
    LicenseMetadata,
    ModelProvenance,
    PackageProvenance,
    Profile,
    ProviderPolicy,
    ProviderProvenance,
    ProviderResult,
)
from .mfa import MfaProvider, parse_textgrid
from .registry import (
    ProfilePreflightError,
    ProfilePreflightReport,
    ProviderRegistry,
    default_registry,
)
from .whisperx import WhisperXProvider

__all__ = [
    "CHORD_VOCABULARY",
    "AudioAnalysisProvider",
    "BeatThisProvider",
    "BrouhahaProvider",
    "CLAPProvider",
    "CapabilityState",
    "CapabilityStatus",
    "ClapProvider",
    "LicenseMetadata",
    "MfaProvider",
    "ModelProvenance",
    "PackageProvenance",
    "Profile",
    "ProfilePreflightError",
    "ProfilePreflightReport",
    "ProviderPolicy",
    "ProviderProvenance",
    "ProviderRegistry",
    "ProviderResult",
    "WhisperXProvider",
    "default_registry",
    "parse_textgrid",
]
