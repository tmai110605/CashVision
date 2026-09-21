# Protocol: Specimen-Disjoint Partitioning and Leave-Specimen-Out (LSO) Evaluation

**Document Identifier:** `protocols/leave_specimen_out_protocol.md`  
**Associated Manuscript:** *Seeing Through the Glare: A Real-Time, Energy-Efficient Mobile Banknote Inspector for Visually Impaired Assistance* (System: CashVision)  
**Target Venue:** *Expert Systems with Applications* (ESWA), Elsevier  
**Scope:** Addressing Reviewer Defect 5.2 (Specimen-Level Data Leakage vs. True Unseen-Note Generalization)

---

## 1. Problem Statement: The Specimen-Level Leakage Defect

In the present manuscript, two distinct experimental protocols were reported:
1. **Protocol A (Section 3.1):** Conducted on benign indoor captures ($600$ images). Because continuous photoshoot frames of the same physical banknote were randomly split into training ($80\%$) and validation ($20\%$), the same physical notes appeared in both train and validation folds. This explains the artificially high baseline recognition ceiling ($99.67\%$--$100.0\%$).
2. **Protocol B (Section 3.1 & 3.3):** The $552$ canonical validation/test images share physical banknote specimens with the canonical training partition ($1{,}260$ images). Furthermore, the $210$ clean-torn notes ($\texttt{torn\_clean}$) and $210$ glare-torn notes ($\texttt{torn\_bright}$) evaluate the exact same physical damaged banknotes photographed under neutral versus directional specular illumination.

While Section 3.1 transparently disclosed this specimen overlap, **disclosure does not repair the scientific measurement**: Tables 2 and 3 measure *photometric, environmental, and illumination invariance across identical physical currency specimens*, rather than *generalization to completely unseen physical banknote instances*.

To provide rigorous empirical evidence of generalization to unseen physical currency, this protocol establishes a **specimen-disjoint partition** constructible directly from the existing $1{,}812$-image static repository without requiring any new data collection.

---

## 2. Dataset Specimen Decomposition without New Collection

Across all six circulating Vietnamese Dong denominations ($10\text{k}, 20\text{k}, 50\text{k}, 100\text{k}, 200\text{k}, 500\text{k}$ VND), the $1{,}812$ static captures were photographed across discrete continuous photoshoot sessions representing identifiable physical banknote specimens:
- Each denomination contains $302$ total images:
  * Intact notes: $232$ images ($65$ indoor, $65$ outdoor, $51$ backlight, $51$ overexposed).
  * Damaged notes: $70$ images ($35$ clean tears, $35$ glare-degraded tears).

### 2.1 The Paired-Tear Specimen Invariant
The $35$ images of $\texttt{torn\_clean}$ and $35$ images of $\texttt{torn\_bright}$ per denomination represent optical pairs of the same physical damaged specimens ($N_{\text{torn\_specimens}} \ge 5$ distinct damaged notes per denomination, each captured across multiple hand orientations).
- **Rule of Non-Contamination:** The $\texttt{torn\_clean}$ and $\texttt{torn\_bright}$ captures for any given physical note $k$ **MUST ALWAYS** reside in the same data split. Under no circumstances may Note $k$'s clean capture be placed in training while Note $k$'s glare capture is placed in testing.

### 2.2 Discrete Specimen Grouping
By auditing photoshoot frame sequences, serial numbering, distinctive fold/crease topography, and tear boundary morphology, the captures for each denomination are grouped into $K = 13$ discrete Physical Specimen Clusters ($K_{\text{total}} = 78$ physical circulating notes across all six denominations):
- **Intact Note Specimens:** $\mathcal{S}_{\text{intact}} = \{S_1, S_2, \dots, S_8\}$ ($M = 8$ distinct physical specimens per denomination, $232$ images total, averaging $29.0$ images/specimen).
- **Damaged Note Specimens:** $\mathcal{S}_{\text{torn}} = \{T_1, T_2, \dots, T_5\}$ ($K_{\text{torn}} = 5$ distinct physical damaged specimens per denomination, $70$ images total, comprising $7$ neutral and $7$ specular-glare images per specimen).

---

## 3. Specimen-Disjoint Partitioning Schemes and Sample Allocations

Two rigorous specimen-disjoint schemes are constructible from this grouping:

### Scheme 1: Canonical Specimen-Disjoint Split (70 / 15 / 15)
Physical specimens within each denomination are partitioned at the specimen level into mutually exclusive sets:
- **Training Pool (70% of specimens):** All frames corresponding to specimens $S_1 \dots S_6$ and $T_1 \dots T_3$ ($K_{\text{train}} = 54$ physical notes across 6 denominations; $1{,}296$ static images, $71.52\%$).
- **Validation Pool (15% of specimens):** All frames corresponding to specimens $S_7$ and $T_4$ ($K_{\text{val}} = 12$ physical notes; $258$ static images, $14.24\%$, used strictly for model checkpoint selection and early stopping).
- **Locked Test Pool (15% of specimens):** All frames corresponding to specimens $S_8$ and $T_5$ ($K_{\text{test}} = 12$ completely unseen physical notes; $258$ static images, $14.24\%$, evaluated across all six operational conditions: $n=54$ indoor, $n=54$ outdoor, $n=42$ backlight, $n=42$ overexposed, $n=33$ clean torn, $n=33$ bright torn).

Zero physical specimens overlap between Training, Validation, and Testing ($\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$).

### Scheme 2: 5-Fold Leave-Specimen-Out (LSO) Cross-Validation
For comprehensive statistical power, the $K_{\text{total}} = 78$ physical specimens are partitioned into 5 disjoint specimen clusters ($\approx 15$--$16$ physical specimens and $\approx 362$ evaluation images per fold):
- In each fold $k \in \{1, \dots, 5\}$, Fold $k$'s physical specimens are held out entirely for testing, while the remaining 4 folds' specimens are used for training and validation.
- All evaluation metrics are aggregated as $\text{mean} \pm \text{std}$ across all 5 folds.

---

## 4. Evaluation Tasks, Generalization Metrics, and Reporting Skeletons

Under the specimen-disjoint split, models must be evaluated on the identical two assistive tasks:
1. **Task 1: Denomination Recognition & Localization on Unseen Banknotes** ($\text{Acc}_{\text{denom}}$, Loc@50, $\overline{\text{IoU}}$).
2. **Task 2: Physical Defect Inspection on Unseen Banknotes** (Tear mAP@50, Binary Tear Accuracy, False Alarm Rate).
3. **Specimen Generalization Gap:** $\Delta_{\text{specimen}} = \text{Acc}_{\text{in-dist}} - \text{Acc}_{\text{unseen}}$.
4. **Specular Defect Degradation Gap:** $\Delta_{\text{tear}} = \text{mAP50}_{\text{clean}} - \text{mAP50}_{\text{bright}}$.
5. **Statistical Significance Testing:** Paired non-parametric Wilcoxon signed-rank test across specimen clusters ($\alpha = 0.05$) and $95\%$ bootstrap confidence intervals ($B = 1{,}000$).

The resulting empirical table skeleton must be added to Section 3 of the manuscript as `Table~\ref{tab:specimen_disjoint_results}` with sample sizes annotated and performance cells marked `---` with `\AUTHORACTION{SPECIMEN-DISJOINT EXPERIMENT REQUIRED: Cluster the 1,812 benchmark images into physical specimen IDs following protocols/leave_specimen_out_protocol.md, execute specimen-disjoint training and inference across YOLOv8n and YOLO11n, and populate Table~\ref{tab:specimen_disjoint_results} to report true generalization to unseen physical banknote specimens.}`.
