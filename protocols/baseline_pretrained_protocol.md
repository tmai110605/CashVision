# Protocol: Pretrained Deep Enhancement Baseline Evaluation and Fine-Tuning

**Document Identifier:** `protocols/baseline_pretrained_protocol.md`  
**Associated Manuscript:** *Seeing Through the Glare: A Real-Time, Energy-Efficient Mobile Banknote Inspector for Visually Impaired Assistance* (System: CashVision)  
**Target Venue:** *Expert Systems with Applications* (ESWA), Elsevier  
**Scope:** Addressing Reviewer Defect 5.1 (Crippled From-Scratch Baselines vs. Pretrained Edge Evaluation)

---

## 1. Problem Statement & Motivation

In the initial submission (Table 3 / `tab:protocol_b_enhancers`), high-capacity deep enhancement models trained from scratch on our domain-constrained augmented dataset exhibited catastrophic performance collapse:
- **IAT (Transformer)** \cite{Cui2022}: $26.94\%$ mean denomination accuracy ($91\text{k}$ parameters)
- **Afifi et al.** \cite{Afifi2021}: $36.23\%$ mean accuracy ($3.29\text{M}$ parameters)
- **EnlightenGAN** \cite{Jiang2021}: $48.28\%$ mean accuracy ($10.17\text{M}$ parameters)
- **RetinexNet** \cite{Wei2018}: $54.69\%$ mean accuracy ($680\text{k}$ parameters)

By comparison, the unenhanced baseline attained $74.42\%$ on YOLOv8n, and classical Gamma correction reached $77.99\%$.

A published state-of-the-art method performing substantially worse than doing nothing indicates a broken training setup, not a scientific finding. Training a $10.2\text{M}$-parameter GAN generator or a multi-scale Laplacian pyramid network from scratch on only $2{,}040$ synthetically augmented instances (derived from $336$ raw captures per fold) without pretraining starves the high-dimensional parameter space of visual priors, causing severe chromatic drift and optimization failure. Presenting this from-scratch collapse as evidence of MQTone's architectural superiority constitutes a straw-man comparison.

To establish a methodologically sound comparison, each baseline must be initialized from its official public pretrained checkpoint and fine-tuned under the identical Protocol B regime. The manuscript must report both regimes side by side:
1. **From-Scratch Regime:** Quantifies sample- and parameter-efficiency in edge-only training from cold start.
2. **Pretrained + Fine-Tuned Regime:** Quantifies peak task-level restoration capability when leveraging large-scale visual priors.

---

## 2. Baseline Model Specifications & Checkpoint Provenance

All deep enhancement baselines must be initialized from official public model weights published by the original authors:

| Baseline Model | Original Publication | Parameter Count | Official Pretrained Weight Source / Dataset | Repository / Checkpoint Access |
|---|---|---|---|---|
| **IAT (Illumination-Adaptive Transformer)** | \citet{Cui2022}, *CVPR 2022* | 91,154 | Pretrained on LOL-v1 / LOL-v2 and MIT-Adobe FiveK exposure benchmarks | `IAT_enhance/` (official PyTorch checkpoint `IAT_LOL.pth` / `IAT_FiveK.pth`) |
| **Afifi et al. (Multi-Scale Exposure Correction)** | \citet{Afifi2021}, *CVPR 2021* | 3,291,273 | Pretrained on the 24,330 multi-scale exposure paired dataset (coarse-to-fine Laplacian pyramid) | Local repository weight `afifi_weights.pt` / official GitHub (`mahmoudnafifi/Exposure_Correction`) |
| **EnlightenGAN** | \citet{Jiang2021}, *IEEE TIP 2021* | 10,165,763 | Pretrained on 1,000 paired/unpaired real low-light/normal-light photograph captures | Local repository weight `enlightengan_weights.pt` / official GitHub (`VITA-Group/EnlightenGAN`) |
| **RetinexNet (Enhance-Net)** | \citet{Wei2018}, *BMVC 2018* | 680,259 | Pretrained on LOL synthetic + real paired dataset | Local repository weight `retinexnet_weights.pt` / official repository |
| **Zero-DCE** | \citet{Guo2020}, *CVPR 2020* | 79,416 | Pretrained zero-reference curve estimation weights | Official GitHub (`Li-Chongyi/Zero-DCE`) |
| **Zero-DCE++** | \citet{Li2021}, *IEEE TPAMI 2021* | 10,561 | Pretrained depthwise separable curve weights | Official GitHub (`Li-Chongyi/Zero-DCE_extension`) |

---

## 3. Fine-Tuning Execution Protocol under Protocol B

To maintain strict experimental parity with MQTone and the unenhanced baseline, the fine-tuning of all pretrained baselines must adhere to the exact Protocol B experimental structure:

### 3.1 Data Splits & Partitioning
- **5-Fold Stratified Cross-Validation:** Identical five folds defined in `train_c2.py` and `combined_metadata_augmented.csv`.
- **Training Set:** Exactly $2{,}040$ augmented training instances per fold ($336$ pristine progenitors $\times$ 5 augmented variations $+$ $120$ oversampled damaged notes).
- **Validation Monitoring Set:** Exactly $696$ images per fold ($570$ augmented validation instances $+$ $126$ real adverse captures from the Generalization Development Set) used for early stopping and best-checkpoint selection.
- **Holdout Evaluation Benchmark:** Identical $1{,}266$-image locked test set spanning all six operational environments (diffuse indoor, outdoor daylight, strong backlighting, severe overexposure, clean physical tears, and torn bright glare).

### 3.2 Optimization & Layer Freezing Strategy
Fine-tuning high-capacity networks on $2{,}040$ instances requires careful learning-rate modulation to avoid catastrophic forgetting of pretrained edge and color priors:
1. **Backbone Feature Preservation (Phase 1 Warmup):**
   - For epochs 1--10: Freeze deep convolutional feature extraction layers; train only the output projection / curve-generation layers using AdamW with $\text{lr} = 10^{-4}$.
   - For epochs 11--100: Unfreeze all layers and fine-tune end-to-end with cosine annealing from $\text{lr}_{\text{max}} = 10^{-4}$ down to $\text{lr}_{\text{min}} = 10^{-6}$.
2. **Coupled Objective:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda_{\text{photo}} \| I_{\text{corr}} - I_{\text{clean}} \|_1$$
   with $\lambda_{\text{photo}} = 1.0$, identical to MQTone's training loss.
3. **Detector Backbone Initialization:**
   - Downstream detector (YOLOv8n / YOLO11n) is initialized from COCO pretrained weights (`yolov8n.pt` / `yolo11n.pt`) with identical detection loss weighting ($\lambda_{\text{cls}} = 0.5, \lambda_{\text{box}} = 7.5, \lambda_{\text{dfl}} = 1.5$).

---

## 4. Reporting Structure in the Manuscript

The resulting empirical metrics must be incorporated into Table 3 (`tab:protocol_b_enhancers`) reporting both regimes side by side:
- **From Scratch:** Existing empirical data (measuring sample efficiency under cold-start edge constraints).
- **Pretrained + Fine-Tuned:** New empirical data populated from this protocol (measuring upper-bound accuracy with domain pretraining).

Where pretrained fine-tuning runs have not yet completed execution, table cells must display `---` accompanied by `\AUTHORACTION{Execute baseline pretrained fine-tuning protocol}`.
