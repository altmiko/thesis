"""Authored feature registry for CICIDS2017 (DistriNet-corrected CICFlowMeter).

This is the *authored* half of the schema profile: plain-English semantics,
category grouping and units for every one of the 79 modelling features. The
*inferred* half (value_type / integer / binary / constant / categorical domain)
is computed from the train split by ``infer_schema.py`` and merged in.

Where the exact CICFlowMeter semantics for a feature are uncertain in this
DistriNet release (bulk / subflow families), the entry is marked
``uncertain: True`` and says so explicitly instead of inventing a description.

Fields per feature:
    category    -- feature family (used for grouping + guiding the miner)
    description -- plain-English meaning
    units       -- physical unit where known, else null
    derived     -- True if computed by the extractor from other features
    uncertain   -- True if semantics/relationships are not reliably established
"""
from __future__ import annotations

# category, description, units, derived, uncertain
CICIDS2017_REGISTRY: dict[str, dict] = {
    # ---- flow identity / context ----
    "Src Port": dict(category="flow_identity", units="port_number", derived=False,
        description="Source TCP/UDP port number of the flow (0-65535)."),
    "Dst Port": dict(category="flow_identity", units="port_number", derived=False,
        description="Destination TCP/UDP port number of the flow (0-65535)."),
    "Protocol": dict(category="flow_identity", units="ip_protocol_number", derived=False,
        description="IP protocol number of the flow (e.g. 6=TCP, 17=UDP, 0=other/unset)."),

    # ---- timing ----
    "Flow Duration": dict(category="timing", units="microseconds", derived=True,
        description="Elapsed time between the first and last packet of the flow."),

    # ---- packet counts ----
    "Total Fwd Packet": dict(category="packet_count", units="packets", derived=True,
        description="Number of packets sent in the forward (client->server) direction."),
    "Total Bwd packets": dict(category="packet_count", units="packets", derived=True,
        description="Number of packets sent in the backward (server->client) direction."),
    "Fwd Act Data Pkts": dict(category="packet_count", units="packets", derived=True,
        description="Number of forward packets carrying at least one byte of TCP payload."),

    # ---- byte counts ----
    "Total Length of Fwd Packet": dict(category="byte_count", units="bytes", derived=True,
        description="Total payload/packet bytes summed over all forward packets."),
    "Total Length of Bwd Packet": dict(category="byte_count", units="bytes", derived=True,
        description="Total payload/packet bytes summed over all backward packets."),

    # ---- packet sizes (directional) ----
    "Fwd Packet Length Max": dict(category="packet_size", units="bytes", derived=True,
        description="Largest forward-packet length in the flow."),
    "Fwd Packet Length Min": dict(category="packet_size", units="bytes", derived=True,
        description="Smallest forward-packet length in the flow."),
    "Fwd Packet Length Mean": dict(category="packet_size", units="bytes", derived=True,
        description="Mean forward-packet length in the flow."),
    "Fwd Packet Length Std": dict(category="packet_size", units="bytes", derived=True,
        description="Standard deviation of forward-packet lengths in the flow."),
    "Bwd Packet Length Max": dict(category="packet_size", units="bytes", derived=True,
        description="Largest backward-packet length in the flow."),
    "Bwd Packet Length Min": dict(category="packet_size", units="bytes", derived=True,
        description="Smallest backward-packet length in the flow."),
    "Bwd Packet Length Mean": dict(category="packet_size", units="bytes", derived=True,
        description="Mean backward-packet length in the flow."),
    "Bwd Packet Length Std": dict(category="packet_size", units="bytes", derived=True,
        description="Standard deviation of backward-packet lengths in the flow."),

    # ---- packet sizes (combined) ----
    "Packet Length Min": dict(category="packet_size", units="bytes", derived=True,
        description="Smallest packet length over both directions of the flow."),
    "Packet Length Max": dict(category="packet_size", units="bytes", derived=True,
        description="Largest packet length over both directions of the flow."),
    "Packet Length Mean": dict(category="packet_size", units="bytes", derived=True,
        description="Mean packet length over both directions of the flow."),
    "Packet Length Std": dict(category="packet_size", units="bytes", derived=True,
        description="Standard deviation of packet length over both directions."),
    "Packet Length Variance": dict(category="packet_size", units="bytes_squared", derived=True,
        description="Variance of packet length over both directions (= Packet Length Std squared)."),
    "Average Packet Size": dict(category="packet_size", units="bytes", derived=True,
        description="Average packet size of the flow (equals Packet Length Mean in this release)."),
    "Fwd Segment Size Avg": dict(category="packet_size", units="bytes", derived=True,
        description="Average forward TCP segment size (equals Fwd Packet Length Mean in this release)."),
    "Bwd Segment Size Avg": dict(category="packet_size", units="bytes", derived=True,
        description="Average backward TCP segment size (equals Bwd Packet Length Mean in this release)."),

    # ---- rates ----
    "Flow Bytes/s": dict(category="rate", units="bytes_per_second", derived=True,
        description="Total flow bytes divided by flow duration in seconds."),
    "Flow Packets/s": dict(category="rate", units="packets_per_second", derived=True,
        description="Total flow packets per second (= Fwd Packets/s + Bwd Packets/s)."),
    "Fwd Packets/s": dict(category="rate", units="packets_per_second", derived=True,
        description="Forward packets per second across the flow duration."),
    "Bwd Packets/s": dict(category="rate", units="packets_per_second", derived=True,
        description="Backward packets per second across the flow duration."),

    # ---- inter-arrival times ----
    "Flow IAT Mean": dict(category="iat", units="microseconds", derived=True,
        description="Mean inter-arrival time between consecutive packets of the flow."),
    "Flow IAT Std": dict(category="iat", units="microseconds", derived=True,
        description="Std of inter-arrival time between consecutive flow packets."),
    "Flow IAT Max": dict(category="iat", units="microseconds", derived=True,
        description="Maximum inter-arrival time between consecutive flow packets."),
    "Flow IAT Min": dict(category="iat", units="microseconds", derived=True,
        description="Minimum inter-arrival time between consecutive flow packets."),
    "Fwd IAT Total": dict(category="iat", units="microseconds", derived=True,
        description="Total time between forward packets (sum of forward inter-arrival gaps)."),
    "Fwd IAT Mean": dict(category="iat", units="microseconds", derived=True,
        description="Mean inter-arrival time between forward packets."),
    "Fwd IAT Std": dict(category="iat", units="microseconds", derived=True,
        description="Std of inter-arrival time between forward packets."),
    "Fwd IAT Max": dict(category="iat", units="microseconds", derived=True,
        description="Maximum inter-arrival time between forward packets."),
    "Fwd IAT Min": dict(category="iat", units="microseconds", derived=True,
        description="Minimum inter-arrival time between forward packets."),
    "Bwd IAT Total": dict(category="iat", units="microseconds", derived=True,
        description="Total time between backward packets (sum of backward inter-arrival gaps)."),
    "Bwd IAT Mean": dict(category="iat", units="microseconds", derived=True,
        description="Mean inter-arrival time between backward packets."),
    "Bwd IAT Std": dict(category="iat", units="microseconds", derived=True,
        description="Std of inter-arrival time between backward packets."),
    "Bwd IAT Max": dict(category="iat", units="microseconds", derived=True,
        description="Maximum inter-arrival time between backward packets."),
    "Bwd IAT Min": dict(category="iat", units="microseconds", derived=True,
        description="Minimum inter-arrival time between backward packets."),

    # ---- TCP flags ----
    "Fwd PSH Flags": dict(category="tcp_flags", units="count", derived=True,
        description="Number of forward packets with the TCP PSH flag set."),
    "Bwd PSH Flags": dict(category="tcp_flags", units="count", derived=True,
        description="Number of backward packets with the TCP PSH flag set."),
    "Fwd URG Flags": dict(category="tcp_flags", units="count", derived=True,
        description="Number of forward packets with the TCP URG flag set (constant 0 in train)."),
    "Bwd URG Flags": dict(category="tcp_flags", units="count", derived=True,
        description="Number of backward packets with the TCP URG flag set (constant 0 in train)."),
    "FIN Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP FIN flag set."),
    "SYN Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP SYN flag set."),
    "RST Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP RST flag set."),
    "PSH Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP PSH flag set."),
    "ACK Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP ACK flag set."),
    "URG Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP URG flag set (constant 0 in train)."),
    "CWR Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP CWR flag set."),
    "ECE Flag Count": dict(category="tcp_flags", units="count", derived=True,
        description="Number of packets in the flow with the TCP ECE flag set."),

    # ---- header statistics ----
    "Fwd Header Length": dict(category="header_stats", units="bytes", derived=True,
        description="Total bytes used for headers in the forward direction."),
    "Bwd Header Length": dict(category="header_stats", units="bytes", derived=True,
        description="Total bytes used for headers in the backward direction."),
    "Fwd Seg Size Min": dict(category="header_stats", units="bytes", derived=True,
        description="Minimum forward TCP segment/header size observed (TCP header-size codes, e.g. 20,32,40)."),

    # ---- ratios ----
    "Down/Up Ratio": dict(category="ratio", units="ratio", derived=True,
        description="Ratio of download to upload activity for the flow (fractional-valued)."),

    # ---- TCP window / data ----
    "FWD Init Win Bytes": dict(category="window_stats", units="bytes", derived=True,
        description="Bytes in the initial TCP receive window of the forward direction."),
    "Bwd Init Win Bytes": dict(category="window_stats", units="bytes", derived=True,
        description="Bytes in the initial TCP receive window of the backward direction."),

    # ---- bulk statistics (CICFlowMeter state-machine; semantics UNRESOLVED here) ----
    "Fwd Bytes/Bulk Avg": dict(category="bulk_stats", units="bytes", derived=True, uncertain=True,
        description="Average bytes per forward bulk transfer, from the CICFlowMeter bulk state machine. Exact semantics/relationships not reliably established for this DistriNet release."),
    "Fwd Packet/Bulk Avg": dict(category="bulk_stats", units="packets", derived=True, uncertain=True,
        description="Average packets per forward bulk transfer (bulk state machine). Semantics UNRESOLVED for this release."),
    "Fwd Bulk Rate Avg": dict(category="bulk_stats", units="bytes_per_second", derived=True, uncertain=True,
        description="Average forward bulk transfer rate (bulk state machine). Semantics UNRESOLVED for this release."),
    "Bwd Bytes/Bulk Avg": dict(category="bulk_stats", units="bytes", derived=True, uncertain=True,
        description="Average bytes per backward bulk transfer (bulk state machine). Semantics UNRESOLVED for this release."),
    "Bwd Packet/Bulk Avg": dict(category="bulk_stats", units="packets", derived=True, uncertain=True,
        description="Average packets per backward bulk transfer (bulk state machine). Semantics UNRESOLVED for this release."),
    "Bwd Bulk Rate Avg": dict(category="bulk_stats", units="bytes_per_second", derived=True, uncertain=True,
        description="Average backward bulk transfer rate (bulk state machine). Semantics UNRESOLVED for this release."),

    # ---- subflow statistics (semantics UNRESOLVED: not equal to totals in this release) ----
    "Subflow Fwd Packets": dict(category="subflow_stats", units="packets", derived=True, uncertain=True,
        description="Forward subflow packet statistic. In this DistriNet release it does NOT equal Total Fwd Packet (observed values are binary 0/1), so no equality rule is asserted; semantics UNRESOLVED."),
    "Subflow Fwd Bytes": dict(category="subflow_stats", units="bytes", derived=True, uncertain=True,
        description="Forward subflow byte statistic. Does NOT equal Total Length of Fwd Packet in this release; semantics UNRESOLVED."),
    "Subflow Bwd Packets": dict(category="subflow_stats", units="packets", derived=True, uncertain=True,
        description="Backward subflow packet statistic. Does NOT equal Total Bwd packets in this release (observed binary 0/1); semantics UNRESOLVED."),
    "Subflow Bwd Bytes": dict(category="subflow_stats", units="bytes", derived=True, uncertain=True,
        description="Backward subflow byte statistic. Does NOT equal Total Length of Bwd Packet in this release; semantics UNRESOLVED."),

    # ---- active / idle ----
    "Active Mean": dict(category="active_idle", units="microseconds", derived=True,
        description="Mean time the flow was active before going idle."),
    "Active Std": dict(category="active_idle", units="microseconds", derived=True,
        description="Std of active-period durations before idling."),
    "Active Max": dict(category="active_idle", units="microseconds", derived=True,
        description="Maximum active-period duration before idling."),
    "Active Min": dict(category="active_idle", units="microseconds", derived=True,
        description="Minimum active-period duration before idling."),
    "Idle Mean": dict(category="active_idle", units="microseconds", derived=True,
        description="Mean time the flow was idle before becoming active."),
    "Idle Std": dict(category="active_idle", units="microseconds", derived=True,
        description="Std of idle-period durations before reactivating."),
    "Idle Max": dict(category="active_idle", units="microseconds", derived=True,
        description="Maximum idle-period duration before reactivating."),
    "Idle Min": dict(category="active_idle", units="microseconds", derived=True,
        description="Minimum idle-period duration before reactivating."),
}


def registry_for(name: str) -> dict:
    return CICIDS2017_REGISTRY.get(name, dict(category="unknown", units=None, derived=None,
                                              uncertain=True, description="No authored description available."))
