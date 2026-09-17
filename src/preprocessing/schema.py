"""CICIoT2023 dataset schema — the single source of truth for column layout.

This module holds ONLY the immutable data schema of the CIC-shipped CICIoT2023
CSV distribution ("Modified Schema A", 39 features): the feature name order, the
label column, the feature data-type groups used for canonical rounding, and the
34-class -> 8-category map.

It deliberately contains no perturbation / adversarial-robustness taxonomy and
no derived masks. The prior ``feature_groups.py`` mixed those two concerns; the
perturbation policy is being rebuilt separately, so it no longer lives here.

Order matters: the parquet writer records this order in metadata and every
downstream module (preprocessing, VAE, attacks, validator, evaluation) indexes
features by position. Reordering FEATURE_NAMES silently corrupts every
persisted array, the scaler, and every trained checkpoint.
"""

LABEL_COLUMN = 'Label'

# ── Feature name list (39 features, order matches CSV header minus Label) ────

FEATURE_NAMES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Variance',
]

assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), "Duplicate feature names"
assert len(FEATURE_NAMES) == 39

# Expected dataset columns (features + label) in the labelled parquet/csv.
EXPECTED_COLUMNS = FEATURE_NAMES + [LABEL_COLUMN]

# ── Data-type groups (drive canonical rounding in pipeline.clip_round) ───────

# Binary protocol/service indicators, values in {0, 1}.
BINARY_FEATURES = [
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
]

# Integer-valued counters/flags.
INTEGER_FEATURES = [
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'Number',
]

# ── Category mapping (uppercase labels as found in the CSV) ──────────────────

CATEGORY_MAP = {
    # DDoS (12)
    'DDOS-ICMP_FLOOD': 'DDoS', 'DDOS-UDP_FLOOD': 'DDoS',
    'DDOS-TCP_FLOOD': 'DDoS', 'DDOS-PSHACK_FLOOD': 'DDoS',
    'DDOS-SYN_FLOOD': 'DDoS', 'DDOS-RSTFINFLOOD': 'DDoS',
    'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS', 'DDOS-UDP_FRAGMENTATION': 'DDoS',
    'DDOS-ACK_FRAGMENTATION': 'DDoS', 'DDOS-ICMP_FRAGMENTATION': 'DDoS',
    'DDOS-HTTP_FLOOD': 'DDoS', 'DDOS-SLOWLORIS': 'DDoS',
    # DoS (4)
    'DOS-UDP_FLOOD': 'DoS', 'DOS-TCP_FLOOD': 'DoS',
    'DOS-SYN_FLOOD': 'DoS', 'DOS-HTTP_FLOOD': 'DoS',
    # Mirai (3)
    'MIRAI-GREETH_FLOOD': 'Mirai', 'MIRAI-UDPPLAIN': 'Mirai',
    'MIRAI-GREIP_FLOOD': 'Mirai',
    # Benign
    'BENIGN': 'Benign',
    # Spoofing (2)
    'MITM-ARPSPOOFING': 'Spoofing', 'DNS_SPOOFING': 'Spoofing',
    # Recon (5)
    'RECON-PINGSWEEP': 'Recon', 'RECON-OSSCAN': 'Recon',
    'RECON-PORTSCAN': 'Recon', 'RECON-HOSTDISCOVERY': 'Recon',
    'VULNERABILITYSCAN': 'Recon',
    # Web (6)
    'BROWSERHIJACKING': 'Web', 'BACKDOOR_MALWARE': 'Web',
    'XSS': 'Web', 'SQLINJECTION': 'Web',
    'COMMANDINJECTION': 'Web', 'UPLOADING_ATTACK': 'Web',
    # BruteForce (1)
    'DICTIONARYBRUTEFORCE': 'BruteForce',
}

assert len(CATEGORY_MAP) == 34, "CATEGORY_MAP must cover all 34 CICIoT2023 labels"
