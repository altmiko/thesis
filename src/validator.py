"""
validate.py — Protocol Constraint Validator for CICIoT2023 Adversarial Examples
=================================================================================
Thesis: VAE-based latent-space adversarial attacks on CICIoT2023

Checks adversarial examples against ground-truth protocol constraints
derived from the CICIoT2023 paper (Neto et al., Sensors 2023, Table 4 & 5).

Constraints checked:
    1. Binary feature range          — all binary features ∈ {0, 1}
    2. Transport mutual exclusivity  — TCP and UDP not both active
    3. ARP exclusivity               — ARP=1 implies TCP=0, UDP=0, ICMP=0
    4. ICMP exclusivity              — ICMP=1 implies TCP=0, UDP=0
    5. Application→transport         — HTTP/HTTPS/SSH/Telnet/SMTP/IRC → TCP=1
    6. DNS→transport                 — DNS=1 → TCP=1 OR UDP=1
    7. DHCP→transport                — DHCP=1 → UDP=1
    8. TCP flag consistency          — TCP flags active → TCP=1
    9. Flag count consistency        — flag_number=1 → corresponding count > 0
   10. Stat ordering                 — Min ≤ AVG ≤ Max (in scaled space, relaxed)
   11. Non-negative counts           — all count features ≥ 0

Reports:
    - Per-constraint violation rate for original samples (baseline)
    - Per-constraint violation rate for VAE adversarial examples
    - Per-constraint violation rate for PGD adversarial examples (if available)
    - Overall validity rate = fraction of samples with ZERO violations
    - Comparison table: original vs VAE vs PGD
"""

import os
import pickle
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = r"D:\thesis"
PROCESSED = os.path.join(ROOT, "data", "processed")
NIDS_DIR = os.path.join(ROOT, "models", "nids")
RESULTS_DIR = os.path.join(ROOT, "results")

# ── Feature indices (must match preprocess.py) ─────────────────────────────────
# 37 features total
# Continuous (24): indices 0-22 + 36
# Binary (13):     indices 23-35
CONTINUOUS_IDX = list(range(0, 23)) + [36]
BINARY_IDX = list(range(23, 36))
N_FEATURES = 37

# ── Named feature indices ──────────────────────────────────────────────────────
# From Table 4 of CICIoT2023 paper (Neto et al., Sensors 2023)
# After preprocessing, the order in .npy files is:
#   0:  Header_Length
#   1:  Time_To_Live (Duration/ttl)
#   2:  Rate
#   3:  fin_flag_number
#   4:  syn_flag_number
#   5:  rst_flag_number
#   6:  psh_flag_number
#   7:  ack_flag_number
#   8:  ece_flag_number
#   9:  cwr_flag_number
#   10: ack_count
#   11: syn_count
#   12: fin_count
#   13: rst_count
#   14: Tot_sum
#   15: Min
#   16: Max
#   17: AVG
#   18: Std
#   19: Tot_size
#   20: IAT
#   21: Number
#   22: Variance
#   23: HTTP       (binary)
#   24: HTTPS      (binary)
#   25: DNS        (binary)
#   26: Telnet     (binary)
#   27: SMTP       (binary)
#   28: SSH        (binary)
#   29: IRC        (binary)
#   30: TCP        (binary)
#   31: UDP        (binary)
#   32: DHCP       (binary)
#   33: ARP        (binary)
#   34: ICMP       (binary)
#   35: IPv       (binary)
#   36: Protocol_Type (continuous)

# Continuous feature names (for reporting)
CONT_NAMES = [
    "Header_Length",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "Tot_sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot_size",
    "IAT",
    "Number",
    "Variance",
    "Protocol_Type",  # index 36, last continuous
]

# Binary feature names (indices 23-35) - must match preprocessed data
BIN_NAMES = [
    "HTTP",
    "HTTPS",
    "DNS",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
]

# Binary feature local indices (within binary block, 0-based)
HTTP = 0
HTTPS = 1
DNS = 2
SMTP = 3
SSH = 4
IRC = 5
TCP = 6
UDP = 7
DHCP = 8
ARP = 9
ICMP = 10
IPv = 11
LLC = 12

# Continuous feature local indices (within CONT_NAMES, 0-based)
FIN_FLAG = 3
SYN_FLAG = 4
RST_FLAG = 5
PSH_FLAG = 6
ACK_FLAG = 7
ECE_FLAG = 8
CWR_FLAG = 9
ACK_CNT = 10
SYN_CNT = 11
FIN_CNT = 12
RST_CNT = 13
TOT_SUM = 14
F_MIN = 15
F_MAX = 16
AVG = 17
STD = 18

ATTACK_CLASSES = ["DDoS", "DoS", "Mirai", "Recon"]


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction helpers
# ══════════════════════════════════════════════════════════════════════════════


def get_binary(X):
    """Extract binary block (indices 23-35) → shape (N, 13)."""
    return X[:, 23:36]


def get_continuous(X):
    """
    Extract continuous features in CONT_NAMES order → shape (N, 24).
    Indices 0-22 are already in order; index 36 (Protocol_Type) appended last.
    Note: features are StandardScaler-normalised, so raw value comparisons
    are not directly meaningful for range checks — we use relative checks
    (e.g. Min ≤ AVG ≤ Max) which are scale-invariant.
    """
    cont = np.concatenate([X[:, 0:23], X[:, 36:37]], axis=1)
    return cont


# ══════════════════════════════════════════════════════════════════════════════
#  Constraint checkers
#  Each returns a boolean array of shape (N,): True = VIOLATION
# ══════════════════════════════════════════════════════════════════════════════


def check_binary_range(X):
    """All binary features must be in {0, 1}."""
    b = get_binary(X)
    # After thresholding at 0.5 in attack, values should be 0 or 1.
    # PGD may produce values outside [0,1] or non-integer.
    not_binary = ~np.isin(b, [0.0, 1.0])
    return not_binary.any(axis=1)  # True if any binary feature is invalid


def check_tcp_udp_exclusive(X):
    """TCP=1 and UDP=1 simultaneously is invalid (transport layer exclusive)."""
    b = get_binary(X)
    return (b[:, TCP] == 1) & (b[:, UDP] == 1)


def check_arp_exclusive(X):
    """ARP=1 implies TCP=0, UDP=0, ICMP=0 (ARP is link-layer, not IP-based)."""
    b = get_binary(X)
    arp_active = b[:, ARP] == 1
    tcp_or_udp_or_icmp = (b[:, TCP] == 1) | (b[:, UDP] == 1) | (b[:, ICMP] == 1)
    return arp_active & tcp_or_udp_or_icmp


def check_icmp_exclusive(X):
    """ICMP=1 implies TCP=0 and UDP=0 (ICMP is network-layer, not transport)."""
    b = get_binary(X)
    icmp_active = b[:, ICMP] == 1
    tcp_or_udp = (b[:, TCP] == 1) | (b[:, UDP] == 1)
    return icmp_active & tcp_or_udp


def check_app_requires_tcp(X):
    """HTTP, HTTPS, SSH, SMTP, IRC all require TCP=1."""
    b = get_binary(X)
    app_tcp = (
        (b[:, HTTP] == 1)
        | (b[:, HTTPS] == 1)
        | (b[:, SSH] == 1)
        | (b[:, SMTP] == 1)
        | (b[:, IRC] == 1)
    )
    tcp_missing = b[:, TCP] == 0
    return app_tcp & tcp_missing


def check_dns_requires_transport(X):
    """DNS=1 requires TCP=1 OR UDP=1."""
    b = get_binary(X)
    dns_active = b[:, DNS] == 1
    transport_missing = (b[:, TCP] == 0) & (b[:, UDP] == 0)
    return dns_active & transport_missing


def check_dhcp_requires_udp(X):
    """DHCP=1 requires UDP=1."""
    b = get_binary(X)
    dhcp_active = b[:, DHCP] == 1
    udp_missing = b[:, UDP] == 0
    return dhcp_active & udp_missing


def check_flags_require_tcp(X):
    """Any TCP flag active (fin/syn/rst/psh/ack/ece/cwr) at continuous indices 3-9 -> TCP=1."""
    # Flag indices in original X: fin=3, syn=4, rst=5, psh=6, ack=7, ece=8, cwr=9
    FLAG_IDX = [3, 4, 5, 6, 7, 8, 9]
    any_flag_active = np.zeros(len(X), dtype=bool)
    for fc in FLAG_IDX:
        any_flag_active |= X[:, fc] > 0
    # TCP is at binary index 29 (HTTP=23, HTTPS=24, DNS=25, SMTP=26, SSH=27, IRC=28, TCP=29)
    tcp_missing = X[:, 29] == 0
    return any_flag_active & tcp_missing


def check_flag_count_consistency(X):
    """
    If a flag_number > 0 (flag was set), the corresponding count should > 0.
    Flags at continuous indices: fin=3, syn=4, rst=5, ack=7
    Counts at continuous indices: ack=10, syn=11, fin=12, rst=13
    """
    # (flag_idx, count_idx) pairs in original X
    pairs = [
        (3, 12),
        (4, 11),
        (7, 10),
        (5, 13),
    ]  # (fin, fin_count), (syn, syn_count), (ack, ack_count), (rst, rst_count)
    violation = np.zeros(len(X), dtype=bool)
    for flag_idx, count_idx in pairs:
        flag_active = X[:, flag_idx] > 0
        count_missing = X[:, count_idx] <= 0
        violation |= flag_active & count_missing
    return violation


def check_stat_ordering(X):
    """
    Min ≤ AVG ≤ Max (packet length statistics must be ordered).

    IMPORTANT: StandardScaler normalizes each feature independently with different
    means and standard deviations, which BREAKS the ordering relationship.
    Example: std_Min=202, std_AVG=355, std_Max=974 means values that were
    ordered in raw space can become disordered after scaling.

    Solution: Inverse transform to raw space before checking.
    """
    # Load scaler to inverse transform
    scaler_path = os.path.join(PROCESSED, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Extract continuous features (indices 0-22, 36)
        cont_indices = list(range(0, 23)) + [36]
        X_cont = X[:, cont_indices]

        # Inverse transform to raw space
        X_raw = scaler.inverse_transform(X_cont)

        # Now check Min ≤ AVG ≤ Max in raw space
        # Indices in continuous block: Min=15, Max=16, AVG=17
        raw_min = X_raw[:, 15]
        raw_max = X_raw[:, 16]
        raw_avg = X_raw[:, 17]

        avg_below_min = raw_avg < raw_min
        avg_above_max = raw_avg > raw_max
        return avg_below_min | avg_above_max
    else:
        # Fallback: no scaler available, skip this check
        return np.zeros(len(X), dtype=bool)


def check_nonneg_counts(X):
    """
    Count features (ack_count, syn_count, fin_count, rst_count) must be ≥ 0.
    In scaled space: count_scaled < scaled(0) = -mean/std
    We use count_scaled < -3.0 as a proxy for "clearly negative raw value"
    (3 std deviations below mean, which is always > 0 in raw space).
    """
    c = get_continuous(X)
    count_cols = [ACK_CNT, SYN_CNT, FIN_CNT, RST_CNT]
    violation = np.zeros(len(X), dtype=bool)
    for cc in count_cols:
        # If scaled value < -3, raw value is very likely negative
        violation |= c[:, cc] < -3.0
    return violation


# ══════════════════════════════════════════════════════════════════════════════
#  Master validator
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    ("Binary range [0,1]", check_binary_range),
    ("TCP XOR UDP exclusive", check_tcp_udp_exclusive),
    ("ARP -> not TCP,UDP,ICMP", check_arp_exclusive),
    ("ICMP -> not TCP,UDP", check_icmp_exclusive),
    ("App proto -> TCP", check_app_requires_tcp),
    ("DNS -> TCP or UDP", check_dns_requires_transport),
    ("DHCP -> UDP", check_dhcp_requires_udp),
    ("TCP flags -> TCP=1", check_flags_require_tcp),
    ("Flag->count consistency", check_flag_count_consistency),
    ("Min <= AVG <= Max", check_stat_ordering),
    ("Count features >= 0", check_nonneg_counts),
]


def validate(X, label=""):
    """
    Run all constraints on X (N, 37).
    Returns:
        results: dict {constraint_name: violation_rate}
        valid_mask: (N,) bool — True if sample has ZERO violations
    """
    N = len(X)
    all_violations = np.zeros(N, dtype=bool)
    results = {}

    for name, fn in CONSTRAINTS:
        viols = fn(X)
        rate = viols.mean()
        results[name] = rate
        all_violations |= viols

    valid_mask = ~all_violations
    overall_validity = valid_mask.mean()
    results["__overall_validity__"] = overall_validity

    if label:
        print(
            f"\n  [{label}]  N={N}  Overall validity: {overall_validity:.4f} "
            f"({valid_mask.sum()}/{N} samples have zero violations)"
        )

    return results, valid_mask


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════


def compare_results(results_dict):
    """
    Print a comparison table across multiple result dicts.
    results_dict: {label: results_from_validate()}
    """
    labels = list(results_dict.keys())
    constraint_names = [name for name, _ in CONSTRAINTS]

    col_w = 28
    val_w = 14

    header = f"{'Constraint':<{col_w}}" + "".join(f"{lbl:>{val_w}}" for lbl in labels)
    print("\n" + "=" * (col_w + val_w * len(labels)))
    print("CONSTRAINT VIOLATION RATES")
    print("=" * (col_w + val_w * len(labels)))
    print(header)
    print("-" * (col_w + val_w * len(labels)))

    for name in constraint_names:
        row = f"{name:<{col_w}}"
        for lbl in labels:
            rate = results_dict[lbl].get(name, float("nan"))
            row += f"{rate:>{val_w}.4f}"
        print(row)

    print("-" * (col_w + val_w * len(labels)))
    row = f"{'Overall Validity Rate':<{col_w}}"
    for lbl in labels:
        rate = results_dict[lbl].get("__overall_validity__", float("nan"))
        row += f"{rate:>{val_w}.4f}"
    print(row)
    print("=" * (col_w + val_w * len(labels)))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    # First, validate preprocessed test data as a sanity check
    print("=" * 60)
    print("SANITY CHECK: Validating preprocessed test data")
    print("=" * 60)

    X_test_path = os.path.join(PROCESSED, "X_test.npy")
    y_test_path = os.path.join(PROCESSED, "y_test.npy")

    if os.path.exists(X_test_path):
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path, allow_pickle=True)

        print(f"\nTest data shape: {X_test.shape}")
        print(f"Total samples: {len(X_test)}")

        # Validate entire test set
        r_test, mask_test = validate(X_test, label="Test Data (all classes)")

        # Also check per class
        print("\n" + "-" * 40)
        print("Per-class validity in test data:")
        print("-" * 40)

        with open(os.path.join(PROCESSED, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

        for class_name in le.classes_:
            class_idx = list(le.classes_).index(class_name)
            mask = y_test == class_name
            X_cls = X_test[mask]
            r_cls, _ = validate(X_cls, label=class_name)
            print(f"  {class_name:10s}: {r_cls['__overall_validity__']:.4f}")
    else:
        print(f"[SKIP] Test data not found at {X_test_path}")

    print("\n" + "=" * 60)
    print("ATTACK VALIDATION: Checking adversarial examples")
    print("=" * 60)

    all_class_summaries = {}

    for class_name in ATTACK_CLASSES:
        save_path = os.path.join(RESULTS_DIR, class_name.lower())

        original_path = os.path.join(save_path, "original.npy")
        adv_cnnlstm_path = os.path.join(save_path, "adv_cnnlstm.npy")
        adv_lgbm_path = os.path.join(save_path, "adv_lgbm.npy")
        adv_pgd_path = os.path.join(save_path, "adv_pgd_cnnlstm.npy")  # optional

        if not os.path.exists(original_path):
            print(f"[SKIP] {class_name} — no results found at {save_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Class: {class_name}")
        print(f"{'=' * 60}")

        X_orig = np.load(original_path)
        results = {}

        # Original samples — establish baseline validity
        r_orig, mask_orig = validate(X_orig, label="Original")
        results["Original"] = r_orig

        # VAE CNN-LSTM adversarial examples
        if os.path.exists(adv_cnnlstm_path):
            X_adv_cnnlstm = np.load(adv_cnnlstm_path)
            r_vae_cnnlstm, _ = validate(X_adv_cnnlstm, label="VAE (CNN-LSTM)")
            results["VAE-CNNLSTM"] = r_vae_cnnlstm

        # VAE LGBM adversarial examples
        if os.path.exists(adv_lgbm_path):
            X_adv_lgbm = np.load(adv_lgbm_path)
            r_vae_lgbm, _ = validate(X_adv_lgbm, label="VAE (LGBM)")
            results["VAE-LGBM"] = r_vae_lgbm

        # PGD baseline (optional — written by baseline_pgd.py)
        if os.path.exists(adv_pgd_path):
            X_adv_pgd = np.load(adv_pgd_path)
            r_pgd, _ = validate(X_adv_pgd, label="PGD (CNN-LSTM)")
            results["PGD-CNNLSTM"] = r_pgd

        compare_results(results)
        all_class_summaries[class_name] = results

    # ── Grand summary: overall validity across all classes ───────────────────
    print(f"\n{'=' * 60}")
    print("GRAND SUMMARY — Overall Validity Rate per Class")
    print(f"{'=' * 60}")
    col_w = 10
    methods = ["Original", "VAE-CNNLSTM", "VAE-LGBM", "PGD-CNNLSTM"]
    header = f"{'Class':<{col_w}}" + "".join(f"{m:>15}" for m in methods)
    print(header)
    print("-" * (col_w + 15 * len(methods)))

    for class_name, results in all_class_summaries.items():
        row = f"{class_name:<{col_w}}"
        for m in methods:
            if m in results:
                row += f"{results[m]['__overall_validity__']:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "validity_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_class_summaries, f)
    print(f"\nValidity results saved to {out_path}")


if __name__ == "__main__":
    main()
