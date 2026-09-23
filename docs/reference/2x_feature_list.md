# Feature Lists for Both Datasets

The lists below preserve the exact model-input order used by the repository. Labels, identifiers, timestamps, and provenance fields that are excluded from the model matrix are not included.

## CICIoT2023 — Modified Schema A (39 features)

Canonical source: `src/preprocessing/schema.py::FEATURE_NAMES`.

1. `Header_Length`
2. `Protocol Type`
3. `Time_To_Live`
4. `Rate`
5. `fin_flag_number`
6. `syn_flag_number`
7. `rst_flag_number`
8. `psh_flag_number`
9. `ack_flag_number`
10. `ece_flag_number`
11. `cwr_flag_number`
12. `ack_count`
13. `syn_count`
14. `fin_count`
15. `rst_count`
16. `HTTP`
17. `HTTPS`
18. `DNS`
19. `Telnet`
20. `SMTP`
21. `SSH`
22. `IRC`
23. `TCP`
24. `UDP`
25. `DHCP`
26. `ARP`
27. `ICMP`
28. `IGMP`
29. `IPv`
30. `LLC`
31. `Tot sum`
32. `Min`
33. `Max`
34. `AVG`
35. `Std`
36. `Tot size`
37. `IAT`
38. `Number`
39. `Variance`

## CICIDS2017-DistriNet (79 features)

Canonical source: `data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json::modelling_feature_names`.

1. `Src Port`
2. `Dst Port`
3. `Protocol`
4. `Flow Duration`
5. `Total Fwd Packet`
6. `Total Bwd packets`
7. `Total Length of Fwd Packet`
8. `Total Length of Bwd Packet`
9. `Fwd Packet Length Max`
10. `Fwd Packet Length Min`
11. `Fwd Packet Length Mean`
12. `Fwd Packet Length Std`
13. `Bwd Packet Length Max`
14. `Bwd Packet Length Min`
15. `Bwd Packet Length Mean`
16. `Bwd Packet Length Std`
17. `Flow Bytes/s`
18. `Flow Packets/s`
19. `Flow IAT Mean`
20. `Flow IAT Std`
21. `Flow IAT Max`
22. `Flow IAT Min`
23. `Fwd IAT Total`
24. `Fwd IAT Mean`
25. `Fwd IAT Std`
26. `Fwd IAT Max`
27. `Fwd IAT Min`
28. `Bwd IAT Total`
29. `Bwd IAT Mean`
30. `Bwd IAT Std`
31. `Bwd IAT Max`
32. `Bwd IAT Min`
33. `Fwd PSH Flags`
34. `Bwd PSH Flags`
35. `Fwd URG Flags`
36. `Bwd URG Flags`
37. `Fwd Header Length`
38. `Bwd Header Length`
39. `Fwd Packets/s`
40. `Bwd Packets/s`
41. `Packet Length Min`
42. `Packet Length Max`
43. `Packet Length Mean`
44. `Packet Length Std`
45. `Packet Length Variance`
46. `FIN Flag Count`
47. `SYN Flag Count`
48. `RST Flag Count`
49. `PSH Flag Count`
50. `ACK Flag Count`
51. `URG Flag Count`
52. `CWR Flag Count`
53. `ECE Flag Count`
54. `Down/Up Ratio`
55. `Average Packet Size`
56. `Fwd Segment Size Avg`
57. `Bwd Segment Size Avg`
58. `Fwd Bytes/Bulk Avg`
59. `Fwd Packet/Bulk Avg`
60. `Fwd Bulk Rate Avg`
61. `Bwd Bytes/Bulk Avg`
62. `Bwd Packet/Bulk Avg`
63. `Bwd Bulk Rate Avg`
64. `Subflow Fwd Packets`
65. `Subflow Fwd Bytes`
66. `Subflow Bwd Packets`
67. `Subflow Bwd Bytes`
68. `FWD Init Win Bytes`
69. `Bwd Init Win Bytes`
70. `Fwd Act Data Pkts`
71. `Fwd Seg Size Min`
72. `Active Mean`
73. `Active Std`
74. `Active Max`
75. `Active Min`
76. `Idle Mean`
77. `Idle Std`
78. `Idle Max`
79. `Idle Min`
