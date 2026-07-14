"""Small source-connector boundary for remote ingestion.

Connectors separate URL selection from extraction outcomes.  The generic web
connector deliberately delegates to the established web extraction pipeline;
it adds no alternate web strategy or provider.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from urllib.parse import ParseResult, urlparse

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.url_safety import validate_remote_url

__all__ = [
    "CallableSourceConnector",
    "ConnectorFailure",
    "ConnectorFailureKind",
    "ConnectorQuality",
    "GenericWebConnector",
    "SourceConnector",
    "SourceConnectorDispatcher",
    "SourceConnectorResult",
]


class ConnectorQuality(StrEnum):
    """The usable quality level reported by a connector."""

    ACCEPTED = "accepted"
    DEGRADED = "degraded"


class ConnectorFailureKind(StrEnum):
    """Machine-readable extraction outcomes that do not yield a source."""

    INVALID_URL = "invalid_url"
    NOT_APPLICABLE = "not_applicable"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass(frozen=True)
class ConnectorFailure:
    """A bounded, typed reason a connector did not yield a document."""

    kind: ConnectorFailureKind
    message: str


@dataclass(frozen=True)
class SourceConnectorResult:
    """A connector's normalized document or a typed failure, never both."""

    source: SourceDoc | None = None
    quality: ConnectorQuality | None = None
    failure: ConnectorFailure | None = None
    connector_name: str = ""

    def __post_init__(self) -> None:
        if (self.source is None) == (self.failure is None):
            raise ValueError("connector result must contain exactly one of source or failure")
        if self.source is not None and self.quality is None:
            raise ValueError("successful connector result requires a quality")
        if self.source is not None and not self.connector_name:
            raise ValueError("successful connector result requires a connector name")
        if self.failure is not None and self.quality is not None:
            raise ValueError("failed connector result cannot have a quality")

    @property
    def succeeded(self) -> bool:
        """Whether this result contains a normalized source document."""
        return self.source is not None


class SourceConnector(Protocol):
    """Minimal contract implemented by every remote source connector."""

    name: str

    def matches(self, parsed: ParseResult, raw_url: str) -> bool:
        """Return whether this connector claims the candidate URL."""

    def extract(self, raw_url: str) -> SourceConnectorResult:
        """Return a normalized source or a typed extraction outcome."""


class CallableSourceConnector:
    """Contract adapter for an already-registered specialist extractor.

    The adapter is intentionally generic: specialist matching and extraction
    stay in their existing modules, while the dispatcher receives uniform
    typed outcomes.
    """

    def __init__(
        self,
        name: str,
        matcher: Callable[[ParseResult, str], bool],
        extractor: Callable[[str], SourceDoc],
        *,
        is_not_applicable: Callable[[Exception], bool] | None = None,
        validated_redirects: bool = False,
    ) -> None:
        self.name = name
        self._matcher = matcher
        self._extractor = extractor
        self._is_not_applicable = is_not_applicable or (lambda _exc: False)
        self.validated_redirects = validated_redirects

    def matches(self, parsed: ParseResult, raw_url: str) -> bool:
        return self._matcher(parsed, raw_url)

    def extract(self, raw_url: str) -> SourceConnectorResult:
        try:
            source = self._extractor(raw_url)
        except Exception as exc:
            kind = (
                ConnectorFailureKind.NOT_APPLICABLE
                if self._is_not_applicable(exc)
                else ConnectorFailureKind.EXTRACTION_FAILED
            )
            return SourceConnectorResult(
                failure=ConnectorFailure(kind, str(exc)[:240]),
                connector_name=self.name,
            )
        if not isinstance(source, SourceDoc):
            return SourceConnectorResult(
                failure=ConnectorFailure(
                    ConnectorFailureKind.EXTRACTION_FAILED,
                    f"{self.name} did not return a SourceDoc",
                ),
                connector_name=self.name,
            )
        return SourceConnectorResult(
            source=source,
            quality=ConnectorQuality.ACCEPTED,
            connector_name=self.name,
        )


class SourceConnectorDispatcher:
    """Dispatch registered specialists before the single generic web connector."""

    def __init__(
        self,
        specialists: list[SourceConnector],
        generic_web: SourceConnector,
    ) -> None:
        self._specialists = tuple(specialists)
        self._generic_web = generic_web

    def dispatch(self, raw_url: str) -> SourceConnectorResult:
        """Route one public URL while preserving specialist fail-closed behavior."""
        try:
            validate_remote_url(raw_url)
        except ValueError as exc:
            return SourceConnectorResult(
                failure=ConnectorFailure(ConnectorFailureKind.INVALID_URL, str(exc)[:240]),
                connector_name="dispatcher",
            )

        parsed = urlparse(raw_url)
        failures: list[tuple[str, ConnectorFailure]] = []
        for connector in self._specialists:
            if not connector.matches(parsed, raw_url):
                continue
            result = connector.extract(raw_url)
            if result.succeeded:
                return result
            assert result.failure is not None
            if result.failure.kind is ConnectorFailureKind.NOT_APPLICABLE:
                continue
            failures.append((connector.name, result.failure))

        if failures:
            details = "; ".join(f"{name}: {failure.message}" for name, failure in failures)
            return SourceConnectorResult(
                failure=ConnectorFailure(ConnectorFailureKind.EXTRACTION_FAILED, details[:240]),
                connector_name="specialist_dispatch",
            )

        if not self._generic_web.matches(parsed, raw_url):
            return SourceConnectorResult(
                failure=ConnectorFailure(
                    ConnectorFailureKind.NOT_APPLICABLE,
                    "no connector accepts this URL",
                ),
                connector_name="dispatcher",
            )
        return self._generic_web.extract(raw_url)


class GenericWebConnector:
    """Validated generic web extraction backed by :func:`extract_web`.

    ``extract_web`` owns its existing bounded streaming and redirect validation
    for every direct HTML fetch.  This connector performs the public URL check
    before delegating so callers have one typed failure surface.
    """

    name = "generic_web"

    def __init__(self, extractor: Callable[[str], SourceDoc] | None = None) -> None:
        self._extractor = extractor

    def matches(self, parsed: ParseResult, raw_url: str) -> bool:
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)

    def extract(self, raw_url: str) -> SourceConnectorResult:
        try:
            validate_remote_url(raw_url)
        except ValueError as exc:
            return SourceConnectorResult(
                failure=ConnectorFailure(ConnectorFailureKind.INVALID_URL, str(exc)[:240]),
                connector_name=self.name,
            )

        extractor = self._extractor
        if extractor is None:
            from obsidian_llm_wiki.ingest.web import extract_web

            extractor = extract_web
        try:
            source = extractor(raw_url)
        except Exception as exc:
            return SourceConnectorResult(
                failure=ConnectorFailure(ConnectorFailureKind.EXTRACTION_FAILED, str(exc)[:240]),
                connector_name=self.name,
            )
        if not isinstance(source, SourceDoc):
            return SourceConnectorResult(
                failure=ConnectorFailure(
                    ConnectorFailureKind.EXTRACTION_FAILED,
                    "web extractor did not return a SourceDoc",
                ),
                connector_name=self.name,
            )
        return SourceConnectorResult(
            source=source,
            quality=_web_quality(source),
            connector_name=self.name,
        )


def _web_quality(source: SourceDoc) -> ConnectorQuality:
    """Map the established extraction diagnostic gate onto the contract type."""
    from obsidian_llm_wiki.ingest.extractors import _check_extraction_quality

    passed, _reason = _check_extraction_quality(source)
    return ConnectorQuality.ACCEPTED if passed else ConnectorQuality.DEGRADED
