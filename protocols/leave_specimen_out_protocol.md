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
By auditing photoshoot frame sequences, serial numbering, and distinctive fold/crease topography, the captures for each denomination are grouped into discrete Physical Specimen Clusters:
- **Intact Note Specimens:** $\mathcal{S}_{\text{intact}} = \{S_1, S_2, \dots, S_M\}$ ($M \ge 8$ distinct physical specimens per denomination).
- **Damaged Note Specimens:** $\mathcal{S}_{\text{torn}} = \{T_1, T_2, \dots, T_K\}$ ($K \ge 5$ distinct physical specimens per denomination).

---

## 3. Specimen-Disjoint Partitioning Schemes

Two rigorous specimen-disjoint schemes are constructible from this grouping:

### Scheme 1: Canonical Specimen-Disjoint Split (70 / 15 / 15)
Physical specimens within each denomination are partitioned at the specimen level:
- **Training Pool (70% of specimens):** All frames corresponding to specimens $S_1 \dots S_6$ and $T_1 \dots T_3$.
- **Validation Pool (15% of specimens):** All frames corresponding to specimen $S_7$ and $T_4$ (used strictly for model selection and early stopping).
- **Locked Test Pool (15% of specimens):** All frames corresponding to specimen $S_8$ and $T_5$ (unseen physical currency evaluated across all environmental conditions).

Zero physical specimens overlap between Training, Validation, and Testing.

### Scheme 2: 5-Fold Leave-Specimen-Out (LSO) Cross-Validation
For comprehensive statistical power, physical specimens per denomination are partitioned into 5 disjoint specimen clusters:
- In each fold $k \in \{1, \dots, 5\}$, Fold $k$'s physical specimens are held out entirely for testing, while the remaining 4 folds' specimens are used for training and validation.
- All evaluation metrics are aggregated as $\text{mean} \pm \text{std}$ across all 5 folds.

---

## 4. Evaluation Tasks and Reporting Skeletons

Under the specimen-disjoint split, models must be evaluated on the identical two assistive tasks:
1. **Task 1: Denomination Recognition & Localization on Unseen Banknotes** ($\text{Acc}_{\text{denom}}$, Loc@50, $\overline{\text{IoU}}$).
2. **Task 2: Physical Defect Inspection on Unseen Banknotes** (Tear mAP@50, Binary Tear Accuracy, False Alarm Rate).
3. **Exact-Match Assistive Reliability** ($\text{Acc}_{\text{exact}}$).

The resulting empirical table skeleton must be added to Section 3 of the manuscript as `Table~\ref{tab:specimen_disjoint_results}` with cells marked `---` and `\AUTHORACTION{Execute specimen-disjoint evaluation protocol}`.
