# CICIDS2017 feature reference (validator_v2)

All 79 DistriNet-corrected CICFlowMeter model features, in schema order, with the automatically inferred type, plain-English meaning, and the validator_v2 rules that reference each. Type facts are inferred on the train split; where exact CICFlowMeter semantics are uncertain in this release the entry says so.

## Flow identity / context

### Src Port  (index 0)

- Source TCP/UDP port number of the flow (0-65535).
- Units: port_number; directly observed; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0001 (`Src Port finite`); SCH_0002 (`Src Port integer`) | PROTOCOL: PROTO_0001 (`Src Port >= 0`)

### Dst Port  (index 1)

- Destination TCP/UDP port number of the flow (0-65535).
- Units: port_number; directly observed; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0003 (`Dst Port finite`); SCH_0004 (`Dst Port integer`) | PROTOCOL: PROTO_0002 (`Dst Port >= 0`)

### Protocol  (index 2)

- IP protocol number of the flow (e.g. 6=TCP, 17=UDP, 0=other/unset).
- Units: ip_protocol_number; directly observed; inferred type: **categorical** domain=[0.0, 6.0, 17.0]
- Validator rules: SCHEMA: SCH_0005 (`Protocol finite`); SCH_0006 (`Protocol in [0.0, 6.0, 17.0]`) | PROTOCOL: PROTO_0003 (`Protocol >= 0`)

## Timing

### Flow Duration  (index 3)

- Elapsed time between the first and last packet of the flow.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0007 (`Flow Duration finite`); SCH_0008 (`Flow Duration integer`) | PROTOCOL: PROTO_0004 (`Flow Duration >= 0`)

## Packet counts

### Total Fwd Packet  (index 4)

- Number of packets sent in the forward (client->server) direction.
- Units: packets; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0009 (`Total Fwd Packet finite`); SCH_0010 (`Total Fwd Packet integer`) | EXTRACTOR: EXT_0006 (`Total Length of Fwd Packet ~= Total Fwd Packet * Fwd Packet Length Mean`) | PROTOCOL: PROTO_0005 (`Total Fwd Packet >= 0`)

### Total Bwd packets  (index 5)

- Number of packets sent in the backward (server->client) direction.
- Units: packets; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0011 (`Total Bwd packets finite`); SCH_0012 (`Total Bwd packets integer`) | EXTRACTOR: EXT_0007 (`Total Length of Bwd Packet ~= Total Bwd packets * Bwd Packet Length Mean`) | PROTOCOL: PROTO_0006 (`Total Bwd packets >= 0`) | MINED: MINED_0010 (`Total Bwd packets == 0  =>  Total Length of Bwd Packet == 0`); MINED_0011 (`Total Bwd packets == 0  =>  Bwd IAT Total == 0`); MINED_0012 (`Total Bwd packets == 0  =>  Bwd IAT Mean == 0`); MINED_0013 (`Total Bwd packets == 0  =>  Bwd IAT Std == 0`) ...

### Fwd Act Data Pkts  (index 69)

- Number of forward packets carrying at least one byte of TCP payload.
- Units: packets; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0118 (`Fwd Act Data Pkts finite`); SCH_0119 (`Fwd Act Data Pkts integer`) | PROTOCOL: PROTO_0070 (`Fwd Act Data Pkts >= 0`)

## Byte counts

### Total Length of Fwd Packet  (index 6)

- Total payload/packet bytes summed over all forward packets.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0013 (`Total Length of Fwd Packet finite`); SCH_0014 (`Total Length of Fwd Packet integer`) | EXTRACTOR: EXT_0006 (`Total Length of Fwd Packet ~= Total Fwd Packet * Fwd Packet Length Mean`) | PROTOCOL: PROTO_0007 (`Total Length of Fwd Packet >= 0`)

### Total Length of Bwd Packet  (index 7)

- Total payload/packet bytes summed over all backward packets.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0015 (`Total Length of Bwd Packet finite`); SCH_0016 (`Total Length of Bwd Packet integer`) | EXTRACTOR: EXT_0007 (`Total Length of Bwd Packet ~= Total Bwd packets * Bwd Packet Length Mean`) | PROTOCOL: PROTO_0008 (`Total Length of Bwd Packet >= 0`) | MINED: MINED_0010 (`Total Bwd packets == 0  =>  Total Length of Bwd Packet == 0`)

## Packet lengths / sizes

### Fwd Packet Length Max  (index 8)

- Largest forward-packet length in the flow.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0017 (`Fwd Packet Length Max finite`); SCH_0018 (`Fwd Packet Length Max integer`) | PROTOCOL: PROTO_0009 (`Fwd Packet Length Max >= 0`) | MINED: MINED_0001 (`Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max`)

### Fwd Packet Length Min  (index 9)

- Smallest forward-packet length in the flow.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0019 (`Fwd Packet Length Min finite`); SCH_0020 (`Fwd Packet Length Min integer`) | PROTOCOL: PROTO_0010 (`Fwd Packet Length Min >= 0`) | MINED: MINED_0001 (`Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max`)

### Fwd Packet Length Mean  (index 10)

- Mean forward-packet length in the flow.
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0021 (`Fwd Packet Length Mean finite`) | EXTRACTOR: EXT_0003 (`Fwd Segment Size Avg ~= Fwd Packet Length Mean`); EXT_0006 (`Total Length of Fwd Packet ~= Total Fwd Packet * Fwd Packet Length Mean`) | PROTOCOL: PROTO_0011 (`Fwd Packet Length Mean >= 0`) | MINED: MINED_0001 (`Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max`)

### Fwd Packet Length Std  (index 11)

- Standard deviation of forward-packet lengths in the flow.
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0022 (`Fwd Packet Length Std finite`) | PROTOCOL: PROTO_0012 (`Fwd Packet Length Std >= 0`)

### Bwd Packet Length Max  (index 12)

- Largest backward-packet length in the flow.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0023 (`Bwd Packet Length Max finite`); SCH_0024 (`Bwd Packet Length Max integer`) | PROTOCOL: PROTO_0013 (`Bwd Packet Length Max >= 0`) | MINED: MINED_0002 (`Bwd Packet Length Min <= Bwd Packet Length Mean <= Bwd Packet Length Max`)

### Bwd Packet Length Min  (index 13)

- Smallest backward-packet length in the flow.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0025 (`Bwd Packet Length Min finite`); SCH_0026 (`Bwd Packet Length Min integer`) | PROTOCOL: PROTO_0014 (`Bwd Packet Length Min >= 0`) | MINED: MINED_0002 (`Bwd Packet Length Min <= Bwd Packet Length Mean <= Bwd Packet Length Max`)

### Bwd Packet Length Mean  (index 14)

- Mean backward-packet length in the flow.
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0027 (`Bwd Packet Length Mean finite`) | EXTRACTOR: EXT_0004 (`Bwd Segment Size Avg ~= Bwd Packet Length Mean`); EXT_0007 (`Total Length of Bwd Packet ~= Total Bwd packets * Bwd Packet Length Mean`) | PROTOCOL: PROTO_0015 (`Bwd Packet Length Mean >= 0`) | MINED: MINED_0002 (`Bwd Packet Length Min <= Bwd Packet Length Mean <= Bwd Packet Length Max`)

### Bwd Packet Length Std  (index 15)

- Standard deviation of backward-packet lengths in the flow.
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0028 (`Bwd Packet Length Std finite`) | PROTOCOL: PROTO_0016 (`Bwd Packet Length Std >= 0`)

### Packet Length Min  (index 40)

- Smallest packet length over both directions of the flow.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0067 (`Packet Length Min finite`); SCH_0068 (`Packet Length Min integer`) | PROTOCOL: PROTO_0041 (`Packet Length Min >= 0`) | MINED: MINED_0006 (`Packet Length Min <= Packet Length Mean <= Packet Length Max`)

### Packet Length Max  (index 41)

- Largest packet length over both directions of the flow.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0069 (`Packet Length Max finite`); SCH_0070 (`Packet Length Max integer`) | PROTOCOL: PROTO_0042 (`Packet Length Max >= 0`) | MINED: MINED_0006 (`Packet Length Min <= Packet Length Mean <= Packet Length Max`)

### Packet Length Mean  (index 42)

- Mean packet length over both directions of the flow.
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0071 (`Packet Length Mean finite`) | EXTRACTOR: EXT_0002 (`Average Packet Size ~= Packet Length Mean`) | PROTOCOL: PROTO_0043 (`Packet Length Mean >= 0`) | MINED: MINED_0006 (`Packet Length Min <= Packet Length Mean <= Packet Length Max`)

### Packet Length Std  (index 43)

- Standard deviation of packet length over both directions.
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0072 (`Packet Length Std finite`) | EXTRACTOR: EXT_0001 (`Packet Length Variance ~= Packet Length Std^2`) | PROTOCOL: PROTO_0044 (`Packet Length Std >= 0`)

### Packet Length Variance  (index 44)

- Variance of packet length over both directions (= Packet Length Std squared).
- Units: bytes_squared; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0073 (`Packet Length Variance finite`) | EXTRACTOR: EXT_0001 (`Packet Length Variance ~= Packet Length Std^2`) | PROTOCOL: PROTO_0045 (`Packet Length Variance >= 0`)

### Average Packet Size  (index 54)

- Average packet size of the flow (equals Packet Length Mean in this release).
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0091 (`Average Packet Size finite`) | EXTRACTOR: EXT_0002 (`Average Packet Size ~= Packet Length Mean`) | PROTOCOL: PROTO_0055 (`Average Packet Size >= 0`)

### Fwd Segment Size Avg  (index 55)

- Average forward TCP segment size (equals Fwd Packet Length Mean in this release).
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0092 (`Fwd Segment Size Avg finite`) | EXTRACTOR: EXT_0003 (`Fwd Segment Size Avg ~= Fwd Packet Length Mean`) | PROTOCOL: PROTO_0056 (`Fwd Segment Size Avg >= 0`)

### Bwd Segment Size Avg  (index 56)

- Average backward TCP segment size (equals Bwd Packet Length Mean in this release).
- Units: bytes; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0093 (`Bwd Segment Size Avg finite`) | EXTRACTOR: EXT_0004 (`Bwd Segment Size Avg ~= Bwd Packet Length Mean`) | PROTOCOL: PROTO_0057 (`Bwd Segment Size Avg >= 0`)

## Rates

### Flow Bytes/s  (index 16)

- Total flow bytes divided by flow duration in seconds.
- Units: bytes_per_second; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0029 (`Flow Bytes/s finite`) | PROTOCOL: PROTO_0017 (`Flow Bytes/s >= 0`)

### Flow Packets/s  (index 17)

- Total flow packets per second (= Fwd Packets/s + Bwd Packets/s).
- Units: packets_per_second; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0030 (`Flow Packets/s finite`) | EXTRACTOR: EXT_0005 (`Flow Packets/s ~= Fwd Packets/s + Bwd Packets/s`) | PROTOCOL: PROTO_0018 (`Flow Packets/s >= 0`)

### Fwd Packets/s  (index 38)

- Forward packets per second across the flow duration.
- Units: packets_per_second; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0065 (`Fwd Packets/s finite`) | EXTRACTOR: EXT_0005 (`Flow Packets/s ~= Fwd Packets/s + Bwd Packets/s`) | PROTOCOL: PROTO_0039 (`Fwd Packets/s >= 0`)

### Bwd Packets/s  (index 39)

- Backward packets per second across the flow duration.
- Units: packets_per_second; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0066 (`Bwd Packets/s finite`) | EXTRACTOR: EXT_0005 (`Flow Packets/s ~= Fwd Packets/s + Bwd Packets/s`) | PROTOCOL: PROTO_0040 (`Bwd Packets/s >= 0`) | MINED: MINED_0016 (`Total Bwd packets == 0  =>  Bwd Packets/s == 0`)

## Inter-arrival times

### Flow IAT Mean  (index 18)

- Mean inter-arrival time between consecutive packets of the flow.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0031 (`Flow IAT Mean finite`) | PROTOCOL: PROTO_0019 (`Flow IAT Mean >= 0`) | MINED: MINED_0003 (`Flow IAT Min <= Flow IAT Mean <= Flow IAT Max`)

### Flow IAT Std  (index 19)

- Std of inter-arrival time between consecutive flow packets.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0032 (`Flow IAT Std finite`) | PROTOCOL: PROTO_0020 (`Flow IAT Std >= 0`)

### Flow IAT Max  (index 20)

- Maximum inter-arrival time between consecutive flow packets.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0033 (`Flow IAT Max finite`); SCH_0034 (`Flow IAT Max integer`) | PROTOCOL: PROTO_0021 (`Flow IAT Max >= 0`) | MINED: MINED_0003 (`Flow IAT Min <= Flow IAT Mean <= Flow IAT Max`)

### Flow IAT Min  (index 21)

- Minimum inter-arrival time between consecutive flow packets.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0035 (`Flow IAT Min finite`); SCH_0036 (`Flow IAT Min integer`) | PROTOCOL: PROTO_0022 (`Flow IAT Min >= 0`) | MINED: MINED_0003 (`Flow IAT Min <= Flow IAT Mean <= Flow IAT Max`)

### Fwd IAT Total  (index 22)

- Total time between forward packets (sum of forward inter-arrival gaps).
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0037 (`Fwd IAT Total finite`); SCH_0038 (`Fwd IAT Total integer`) | PROTOCOL: PROTO_0023 (`Fwd IAT Total >= 0`)

### Fwd IAT Mean  (index 23)

- Mean inter-arrival time between forward packets.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0039 (`Fwd IAT Mean finite`) | PROTOCOL: PROTO_0024 (`Fwd IAT Mean >= 0`) | MINED: MINED_0004 (`Fwd IAT Min <= Fwd IAT Mean <= Fwd IAT Max`)

### Fwd IAT Std  (index 24)

- Std of inter-arrival time between forward packets.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0040 (`Fwd IAT Std finite`) | PROTOCOL: PROTO_0025 (`Fwd IAT Std >= 0`)

### Fwd IAT Max  (index 25)

- Maximum inter-arrival time between forward packets.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0041 (`Fwd IAT Max finite`); SCH_0042 (`Fwd IAT Max integer`) | PROTOCOL: PROTO_0026 (`Fwd IAT Max >= 0`) | MINED: MINED_0004 (`Fwd IAT Min <= Fwd IAT Mean <= Fwd IAT Max`)

### Fwd IAT Min  (index 26)

- Minimum inter-arrival time between forward packets.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0043 (`Fwd IAT Min finite`); SCH_0044 (`Fwd IAT Min integer`) | PROTOCOL: PROTO_0027 (`Fwd IAT Min >= 0`) | MINED: MINED_0004 (`Fwd IAT Min <= Fwd IAT Mean <= Fwd IAT Max`)

### Bwd IAT Total  (index 27)

- Total time between backward packets (sum of backward inter-arrival gaps).
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0045 (`Bwd IAT Total finite`); SCH_0046 (`Bwd IAT Total integer`) | PROTOCOL: PROTO_0028 (`Bwd IAT Total >= 0`) | MINED: MINED_0011 (`Total Bwd packets == 0  =>  Bwd IAT Total == 0`)

### Bwd IAT Mean  (index 28)

- Mean inter-arrival time between backward packets.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0047 (`Bwd IAT Mean finite`) | PROTOCOL: PROTO_0029 (`Bwd IAT Mean >= 0`) | MINED: MINED_0005 (`Bwd IAT Min <= Bwd IAT Mean <= Bwd IAT Max`); MINED_0012 (`Total Bwd packets == 0  =>  Bwd IAT Mean == 0`)

### Bwd IAT Std  (index 29)

- Std of inter-arrival time between backward packets.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0048 (`Bwd IAT Std finite`) | PROTOCOL: PROTO_0030 (`Bwd IAT Std >= 0`) | MINED: MINED_0013 (`Total Bwd packets == 0  =>  Bwd IAT Std == 0`)

### Bwd IAT Max  (index 30)

- Maximum inter-arrival time between backward packets.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0049 (`Bwd IAT Max finite`); SCH_0050 (`Bwd IAT Max integer`) | PROTOCOL: PROTO_0031 (`Bwd IAT Max >= 0`) | MINED: MINED_0005 (`Bwd IAT Min <= Bwd IAT Mean <= Bwd IAT Max`); MINED_0014 (`Total Bwd packets == 0  =>  Bwd IAT Max == 0`)

### Bwd IAT Min  (index 31)

- Minimum inter-arrival time between backward packets.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0051 (`Bwd IAT Min finite`); SCH_0052 (`Bwd IAT Min integer`) | PROTOCOL: PROTO_0032 (`Bwd IAT Min >= 0`) | MINED: MINED_0005 (`Bwd IAT Min <= Bwd IAT Mean <= Bwd IAT Max`); MINED_0015 (`Total Bwd packets == 0  =>  Bwd IAT Min == 0`)

## TCP flags

### Fwd PSH Flags  (index 32)

- Number of forward packets with the TCP PSH flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0053 (`Fwd PSH Flags finite`); SCH_0054 (`Fwd PSH Flags integer`) | PROTOCOL: PROTO_0033 (`Fwd PSH Flags >= 0`) | MINED: MINED_0009 (`PSH Flag Count ~= Fwd PSH Flags + Bwd PSH Flags`)

### Bwd PSH Flags  (index 33)

- Number of backward packets with the TCP PSH flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0055 (`Bwd PSH Flags finite`); SCH_0056 (`Bwd PSH Flags integer`) | PROTOCOL: PROTO_0034 (`Bwd PSH Flags >= 0`) | MINED: MINED_0009 (`PSH Flag Count ~= Fwd PSH Flags + Bwd PSH Flags`)

### Fwd URG Flags  (index 34)

- Number of forward packets with the TCP URG flag set (constant 0 in train).
- Units: count; derived; inferred type: **constant** constant=0.0
- Validator rules: SCHEMA: SCH_0057 (`Fwd URG Flags finite`); SCH_0058 (`Fwd URG Flags == 0.0`) | PROTOCOL: PROTO_0035 (`Fwd URG Flags >= 0`)

### Bwd URG Flags  (index 35)

- Number of backward packets with the TCP URG flag set (constant 0 in train).
- Units: count; derived; inferred type: **constant** constant=0.0
- Validator rules: SCHEMA: SCH_0059 (`Bwd URG Flags finite`); SCH_0060 (`Bwd URG Flags == 0.0`) | PROTOCOL: PROTO_0036 (`Bwd URG Flags >= 0`)

### FIN Flag Count  (index 45)

- Number of packets in the flow with the TCP FIN flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0074 (`FIN Flag Count finite`); SCH_0075 (`FIN Flag Count integer`) | PROTOCOL: PROTO_0046 (`FIN Flag Count >= 0`)

### SYN Flag Count  (index 46)

- Number of packets in the flow with the TCP SYN flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0076 (`SYN Flag Count finite`); SCH_0077 (`SYN Flag Count integer`) | PROTOCOL: PROTO_0047 (`SYN Flag Count >= 0`)

### RST Flag Count  (index 47)

- Number of packets in the flow with the TCP RST flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0078 (`RST Flag Count finite`); SCH_0079 (`RST Flag Count integer`) | PROTOCOL: PROTO_0048 (`RST Flag Count >= 0`)

### PSH Flag Count  (index 48)

- Number of packets in the flow with the TCP PSH flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0080 (`PSH Flag Count finite`); SCH_0081 (`PSH Flag Count integer`) | PROTOCOL: PROTO_0049 (`PSH Flag Count >= 0`) | MINED: MINED_0009 (`PSH Flag Count ~= Fwd PSH Flags + Bwd PSH Flags`)

### ACK Flag Count  (index 49)

- Number of packets in the flow with the TCP ACK flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0082 (`ACK Flag Count finite`); SCH_0083 (`ACK Flag Count integer`) | PROTOCOL: PROTO_0050 (`ACK Flag Count >= 0`)

### URG Flag Count  (index 50)

- Number of packets in the flow with the TCP URG flag set (constant 0 in train).
- Units: count; derived; inferred type: **constant** constant=0.0
- Validator rules: SCHEMA: SCH_0084 (`URG Flag Count finite`); SCH_0085 (`URG Flag Count == 0.0`) | PROTOCOL: PROTO_0051 (`URG Flag Count >= 0`)

### CWR Flag Count  (index 51)

- Number of packets in the flow with the TCP CWR flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0086 (`CWR Flag Count finite`); SCH_0087 (`CWR Flag Count integer`) | PROTOCOL: PROTO_0052 (`CWR Flag Count >= 0`)

### ECE Flag Count  (index 52)

- Number of packets in the flow with the TCP ECE flag set.
- Units: count; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0088 (`ECE Flag Count finite`); SCH_0089 (`ECE Flag Count integer`) | PROTOCOL: PROTO_0053 (`ECE Flag Count >= 0`)

## Header statistics

### Fwd Header Length  (index 36)

- Total bytes used for headers in the forward direction.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0061 (`Fwd Header Length finite`); SCH_0062 (`Fwd Header Length integer`) | PROTOCOL: PROTO_0037 (`Fwd Header Length >= 0`)

### Bwd Header Length  (index 37)

- Total bytes used for headers in the backward direction.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0063 (`Bwd Header Length finite`); SCH_0064 (`Bwd Header Length integer`) | PROTOCOL: PROTO_0038 (`Bwd Header Length >= 0`)

### Fwd Seg Size Min  (index 70)

- Minimum forward TCP segment/header size observed (TCP header-size codes, e.g. 20,32,40).
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0120 (`Fwd Seg Size Min finite`); SCH_0121 (`Fwd Seg Size Min integer`) | PROTOCOL: PROTO_0071 (`Fwd Seg Size Min >= 0`)

## Ratios

### Down/Up Ratio  (index 53)

- Ratio of download to upload activity for the flow (fractional-valued).
- Units: ratio; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0090 (`Down/Up Ratio finite`) | PROTOCOL: PROTO_0054 (`Down/Up Ratio >= 0`)

## TCP window / data

### FWD Init Win Bytes  (index 67)

- Bytes in the initial TCP receive window of the forward direction.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0114 (`FWD Init Win Bytes finite`); SCH_0115 (`FWD Init Win Bytes integer`) | PROTOCOL: PROTO_0068 (`FWD Init Win Bytes >= 0`)

### Bwd Init Win Bytes  (index 68)

- Bytes in the initial TCP receive window of the backward direction.
- Units: bytes; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0116 (`Bwd Init Win Bytes finite`); SCH_0117 (`Bwd Init Win Bytes integer`) | PROTOCOL: PROTO_0069 (`Bwd Init Win Bytes >= 0`)

## Bulk statistics

### Fwd Bytes/Bulk Avg  (index 57)

- Average bytes per forward bulk transfer, from the CICFlowMeter bulk state machine. Exact semantics/relationships not reliably established for this DistriNet release.
- Units: bytes; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0094 (`Fwd Bytes/Bulk Avg finite`); SCH_0095 (`Fwd Bytes/Bulk Avg integer`) | PROTOCOL: PROTO_0058 (`Fwd Bytes/Bulk Avg >= 0`)

### Fwd Packet/Bulk Avg  (index 58)

- Average packets per forward bulk transfer (bulk state machine). Semantics UNRESOLVED for this release.
- Units: packets; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0096 (`Fwd Packet/Bulk Avg finite`); SCH_0097 (`Fwd Packet/Bulk Avg integer`) | PROTOCOL: PROTO_0059 (`Fwd Packet/Bulk Avg >= 0`)

### Fwd Bulk Rate Avg  (index 59)

- Average forward bulk transfer rate (bulk state machine). Semantics UNRESOLVED for this release.
- Units: bytes_per_second; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0098 (`Fwd Bulk Rate Avg finite`); SCH_0099 (`Fwd Bulk Rate Avg integer`) | PROTOCOL: PROTO_0060 (`Fwd Bulk Rate Avg >= 0`)

### Bwd Bytes/Bulk Avg  (index 60)

- Average bytes per backward bulk transfer (bulk state machine). Semantics UNRESOLVED for this release.
- Units: bytes; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0100 (`Bwd Bytes/Bulk Avg finite`); SCH_0101 (`Bwd Bytes/Bulk Avg integer`) | PROTOCOL: PROTO_0061 (`Bwd Bytes/Bulk Avg >= 0`)

### Bwd Packet/Bulk Avg  (index 61)

- Average packets per backward bulk transfer (bulk state machine). Semantics UNRESOLVED for this release.
- Units: packets; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0102 (`Bwd Packet/Bulk Avg finite`); SCH_0103 (`Bwd Packet/Bulk Avg integer`) | PROTOCOL: PROTO_0062 (`Bwd Packet/Bulk Avg >= 0`)

### Bwd Bulk Rate Avg  (index 62)

- Average backward bulk transfer rate (bulk state machine). Semantics UNRESOLVED for this release.
- Units: bytes_per_second; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0104 (`Bwd Bulk Rate Avg finite`); SCH_0105 (`Bwd Bulk Rate Avg integer`) | PROTOCOL: PROTO_0063 (`Bwd Bulk Rate Avg >= 0`)

## Subflow statistics

### Subflow Fwd Packets  (index 63)

- Forward subflow packet statistic. In this DistriNet release it does NOT equal Total Fwd Packet (observed values are binary 0/1), so no equality rule is asserted; semantics UNRESOLVED.
- Units: packets; derived; inferred type: **binary** domain=[0.0, 1.0]
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0106 (`Subflow Fwd Packets finite`); SCH_0107 (`Subflow Fwd Packets in {0,1}`) | PROTOCOL: PROTO_0064 (`Subflow Fwd Packets >= 0`)

### Subflow Fwd Bytes  (index 64)

- Forward subflow byte statistic. Does NOT equal Total Length of Fwd Packet in this release; semantics UNRESOLVED.
- Units: bytes; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0108 (`Subflow Fwd Bytes finite`); SCH_0109 (`Subflow Fwd Bytes integer`) | PROTOCOL: PROTO_0065 (`Subflow Fwd Bytes >= 0`)

### Subflow Bwd Packets  (index 65)

- Backward subflow packet statistic. Does NOT equal Total Bwd packets in this release (observed binary 0/1); semantics UNRESOLVED.
- Units: packets; derived; inferred type: **binary** domain=[0.0, 1.0]
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0110 (`Subflow Bwd Packets finite`); SCH_0111 (`Subflow Bwd Packets in {0,1}`) | PROTOCOL: PROTO_0066 (`Subflow Bwd Packets >= 0`)

### Subflow Bwd Bytes  (index 66)

- Backward subflow byte statistic. Does NOT equal Total Length of Bwd Packet in this release; semantics UNRESOLVED.
- Units: bytes; derived; inferred type: **integer**
- **Uncertain:** exact CICFlowMeter semantics/relationships are not reliably established for this release; no hard relational rule is asserted.
- Validator rules: SCHEMA: SCH_0112 (`Subflow Bwd Bytes finite`); SCH_0113 (`Subflow Bwd Bytes integer`) | PROTOCOL: PROTO_0067 (`Subflow Bwd Bytes >= 0`)

## Active / idle statistics

### Active Mean  (index 71)

- Mean time the flow was active before going idle.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0122 (`Active Mean finite`) | PROTOCOL: PROTO_0072 (`Active Mean >= 0`) | MINED: MINED_0007 (`Active Min <= Active Mean <= Active Max`)

### Active Std  (index 72)

- Std of active-period durations before idling.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0123 (`Active Std finite`) | PROTOCOL: PROTO_0073 (`Active Std >= 0`)

### Active Max  (index 73)

- Maximum active-period duration before idling.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0124 (`Active Max finite`); SCH_0125 (`Active Max integer`) | PROTOCOL: PROTO_0074 (`Active Max >= 0`) | MINED: MINED_0007 (`Active Min <= Active Mean <= Active Max`)

### Active Min  (index 74)

- Minimum active-period duration before idling.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0126 (`Active Min finite`); SCH_0127 (`Active Min integer`) | PROTOCOL: PROTO_0075 (`Active Min >= 0`) | MINED: MINED_0007 (`Active Min <= Active Mean <= Active Max`)

### Idle Mean  (index 75)

- Mean time the flow was idle before becoming active.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0128 (`Idle Mean finite`) | PROTOCOL: PROTO_0076 (`Idle Mean >= 0`) | MINED: MINED_0008 (`Idle Min <= Idle Mean <= Idle Max`)

### Idle Std  (index 76)

- Std of idle-period durations before reactivating.
- Units: microseconds; derived; inferred type: **numeric**
- Validator rules: SCHEMA: SCH_0129 (`Idle Std finite`) | PROTOCOL: PROTO_0077 (`Idle Std >= 0`)

### Idle Max  (index 77)

- Maximum idle-period duration before reactivating.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0130 (`Idle Max finite`); SCH_0131 (`Idle Max integer`) | PROTOCOL: PROTO_0078 (`Idle Max >= 0`) | MINED: MINED_0008 (`Idle Min <= Idle Mean <= Idle Max`)

### Idle Min  (index 78)

- Minimum idle-period duration before reactivating.
- Units: microseconds; derived; inferred type: **integer**
- Validator rules: SCHEMA: SCH_0132 (`Idle Min finite`); SCH_0133 (`Idle Min integer`) | PROTOCOL: PROTO_0079 (`Idle Min >= 0`) | MINED: MINED_0008 (`Idle Min <= Idle Mean <= Idle Max`)
