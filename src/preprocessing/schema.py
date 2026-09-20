"""Canonical CICIoT2023 CSV-level feature schema.

The released CSV rows summarize packet windows (10/100 packets in the dataset
feature-extraction procedure). Packet-level flags, service indicators, counters,
and protocol identifiers are therefore represented by window aggregates. A
packet-level discrete property is not a discrete CSV-level variable: fractional
values are meaningful and MUST be preserved.

This module fixes the immutable 39-column order and descriptive/structural
semantics only. Empirical support, perturbability tiers, and learned constraints
must be fitted on the natural training partition and stored separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

LABEL_COLUMN = "Label"

FEATURE_NAMES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Variance",
]
assert len(FEATURE_NAMES) == 39 and len(set(FEATURE_NAMES)) == 39
EXPECTED_COLUMNS = FEATURE_NAMES + [LABEL_COLUMN]

RepresentationType = Literal[
    "bounded_aggregated_indicator",
    "nonnegative_continuous_aggregate",
    "aggregated_code_like",
    "bounded_continuous_aggregate",
]


@dataclass(frozen=True)
class FeatureSpec:
    """CSV-level semantics, distinct from empirical and attack policies."""

    name: str
    semantic_family: str
    representation_type: RepresentationType
    expected_min: float | None
    expected_max: float | None
    source_semantics: str
    derived: bool
    potential_mutability_class: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FLAG_AGGREGATES = (
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
)
SERVICE_PROTOCOL_AGGREGATES = (
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
)
BOUNDED_AGGREGATED_FEATURES = FLAG_AGGREGATES + SERVICE_PROTOCOL_AGGREGATES
COUNT_AGGREGATES = ("ack_count", "syn_count", "fin_count", "rst_count", "Number")
DERIVED_STATISTICS = (
    "Rate", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Variance",
)
NONNEGATIVE_FEATURES = tuple(FEATURE_NAMES)


def _spec(
    name: str,
    family: str,
    representation: RepresentationType,
    *,
    maximum: float | None = None,
    semantics: str,
    derived: bool,
    notes: str = "",
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        semantic_family=family,
        representation_type=representation,
        expected_min=0.0,
        expected_max=maximum,
        source_semantics=semantics,
        derived=derived,
        potential_mutability_class="must_be_train_mined",
        notes=notes,
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    _spec("Header_Length", "header", "nonnegative_continuous_aggregate",
          semantics="window aggregate of packet header length", derived=False,
          notes="Large values are possible after window aggregation; empirical percentiles are not validity bounds."),
    _spec("Protocol Type", "protocol", "aggregated_code_like", maximum=255.0,
          semantics="window aggregate of numeric packet protocol identifiers", derived=False,
          notes="Not a categorical packet protocol at CSV-row level; fractional averages are expected."),
    _spec("Time_To_Live", "lifetime", "bounded_continuous_aggregate", maximum=255.0,
          semantics="window aggregate of packet TTL values", derived=False,
          notes="A mean of per-packet 8-bit TTL values remains continuous in [0,255]."),
    _spec("Rate", "timing", "nonnegative_continuous_aggregate",
          semantics="derived traffic-rate statistic for the packet window", derived=True),
    *(
        _spec(name, "tcp_flag", "bounded_aggregated_indicator", maximum=1.0,
              semantics="mean occurrence of a packet-level flag within the window", derived=False,
              notes="Fractional values encode occurrence frequency and must not be binarized.")
        for name in FLAG_AGGREGATES
    ),
    *(
        _spec(name, "packet_count", "nonnegative_continuous_aggregate",
              semantics="window-aggregated packet-count statistic", derived=False,
              notes="Fractional values are legitimate window means and must not be rounded.")
        for name in COUNT_AGGREGATES[:-1]
    ),
    *(
        _spec(name, "service_or_protocol", "bounded_aggregated_indicator", maximum=1.0,
              semantics="mean occurrence of a packet-level service/protocol indicator within the window", derived=False,
              notes="Fractional values encode occurrence frequency and must not be binarized.")
        for name in SERVICE_PROTOCOL_AGGREGATES
    ),
    _spec("Tot sum", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived total packet-size statistic for the window", derived=True),
    _spec("Min", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived minimum packet-size statistic for the window", derived=True),
    _spec("Max", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived maximum packet-size statistic for the window", derived=True),
    _spec("AVG", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived mean packet-size statistic for the window", derived=True),
    _spec("Std", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived packet-size standard deviation for the window", derived=True),
    _spec("Tot size", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived packet-size aggregate for the window", derived=True),
    _spec("IAT", "timing", "nonnegative_continuous_aggregate",
          semantics="derived inter-arrival-time statistic emitted by the extractor", derived=True,
          notes="Units must be established from source/code evidence; preprocessing performs no unit conversion."),
    _spec("Number", "packet_count", "nonnegative_continuous_aggregate",
          semantics="window-aggregated count statistic", derived=True,
          notes="Fractional values are legitimate window means and must not be rounded."),
    _spec("Variance", "packet_size", "nonnegative_continuous_aggregate",
          semantics="derived packet-size variance statistic for the window", derived=True),
)

assert tuple(spec.name for spec in FEATURE_SPECS) == tuple(FEATURE_NAMES)
FEATURE_METADATA = {spec.name: spec for spec in FEATURE_SPECS}
BOUNDED_AGGREGATED_IDX = tuple(FEATURE_NAMES.index(name) for name in BOUNDED_AGGREGATED_FEATURES)
NONNEGATIVE_IDX = tuple(range(len(FEATURE_NAMES)))


def feature_metadata_records() -> list[dict[str, object]]:
    """Return JSON-serializable metadata in frozen feature order."""

    return [spec.to_dict() for spec in FEATURE_SPECS]


CATEGORY_MAP = {
    "DDOS-ICMP_FLOOD": "DDoS", "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS", "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS", "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS", "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS", "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS", "DDOS-SLOWLORIS": "DDoS",
    "DOS-UDP_FLOOD": "DoS", "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS", "DOS-HTTP_FLOOD": "DoS",
    "MIRAI-GREETH_FLOOD": "Mirai", "MIRAI-UDPPLAIN": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "BENIGN": "Benign",
    "MITM-ARPSPOOFING": "Spoofing", "DNS_SPOOFING": "Spoofing",
    "RECON-PINGSWEEP": "Recon", "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon", "RECON-HOSTDISCOVERY": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "BROWSERHIJACKING": "Web", "BACKDOOR_MALWARE": "Web",
    "XSS": "Web", "SQLINJECTION": "Web",
    "COMMANDINJECTION": "Web", "UPLOADING_ATTACK": "Web",
    "DICTIONARYBRUTEFORCE": "BruteForce",
}
assert len(CATEGORY_MAP) == 34
