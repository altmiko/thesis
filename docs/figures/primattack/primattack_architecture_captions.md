# PrimAttack architecture figure

Source: `primattack_architecture.drawio` (page 1 = panel a, page 2 = panel b).
Exports: `primattack_architecture_{a,b}.pdf` (cropped vector, for LaTeX) and `.png` (scale 2, editable XML embedded).
Sized for 160 mm print width (body text 8.2 pt, titles 9.1 pt).

## (a) End-to-end pipeline

Each cell fixes the dataset, victim, class, budget, primitive mode, attack seed and optimizer. The per-class caps p_max and r_max, the 9-feature p99 envelope and the DoS/DDoS minimum packet-rate floor are fitted on the train split only, and all methods attack the same 800 frozen test flows. validator_v2 checks only the final x′, given its source flow, in the post-attack evaluation, where every ASR shares the denominator N = 800. The executed FINAL runs also applied this check inside the search, and amendment A6 shows that it changed no final flow. The frozen victim f also decides which test flows count as clean-correct during source selection and scores raw success in the post-attack evaluation; these two edges are not drawn.

## (b) RealizedSearch, per flow

The identity flow is scored first at a cost of 1, and each gradient step costs 2 victim evaluations (one surrogate forward-backward pass, one realized evaluation). The surrogate supplies a straight-through gradient only; only realized, quantized flows can become the incumbent. Among non-hits the incumbent is the lowest-margin candidate, and the search for a flow stops once its remaining budget cannot pay for another step.
