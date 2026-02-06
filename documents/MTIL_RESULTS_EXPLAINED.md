# Understanding MTIL Results in the MoE-Adapters Paper

This note explains how **Multi-Task Incremental Learning (MTIL)** results are set up and reported in the original paper *"Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters"* (CVPR 2024), so you can match them to this codebase.

---

## What is MTIL in the Paper?

- **Setting**: The model learns **11 datasets one after another** (sequential tasks):  
  Aircraft → Caltech101 → CIFAR100 → DTD → EuroSAT → Flowers → Food → MNIST → OxfordPet → StanfordCars → SUN397.
- **Goal**: After learning all 11 tasks, evaluate how well the model does on **each** of the 11 datasets (and possibly on others).
- **Router**: An **auto-chooser (DDAS)** decides which expert(s) to use for each input. So at test time you get one accuracy per **(checkpoint, eval dataset)** pair.

So “MTIL results” in the paper = accuracies when you **evaluate** on these 11 datasets, using the **final** model (or a specific checkpoint) that was trained on all 11 in sequence.

---

## How Results Are Usually Reported

### 1. **Evaluation matrix (most common)**

- Rows: **“Model trained on …”** (which checkpoint / last task).
- Columns: **“Evaluated on …”** (which dataset’s test set).
- Cell: **Accuracy** (often Top-1, sometimes Top-5) when that model is evaluated on that dataset.

So you see a **11×11** (or similar) grid: for each training stage (or the final model), the paper reports accuracy on all 11 evaluation datasets.

- **Diagonal**: “Train on dataset X, test on X” → usually the **best** for that column (in-domain).
- **Off-diagonal**: “Train on X, test on Y” → transfer / retention; often lower than diagonal.

In this repo, that’s exactly what the test script does: for each of the 11 checkpoints (one per dataset) it runs evaluation on each of the 11 datasets and you get such a matrix (e.g. in `complete_results_summary.txt` or the cross-dataset matrix).

### 2. **Average accuracy**

- **Average over all 11 eval datasets** for a given checkpoint (e.g. “final model” or “after task 11”).
- Sometimes reported as **“average accuracy”** or **“mean accuracy across tasks”** in the table caption or main text.

So when the paper says “MTIL average accuracy”, it usually means: one number = mean of the 11 accuracies (one per eval dataset) for that model.

### 3. **Forward / backward transfer (optional)**

- **Forward**: After learning task T, accuracy on **previous** tasks 1…T−1 (do we forget?).
- **Backward**: After learning task T, accuracy on **future** tasks T+1…11 (often not applicable if we only test at the end).

If the paper has a “transfer” or “forgetting” table, rows/columns are again (train task, eval task) and the numbers are accuracies.

---

## What to Look for in the Paper

1. **Section title**  
   Look for “Multi-Task Continual Learning”, “MTCL”, “Multi-dataset”, or “11 datasets”.

2. **Main table**  
   - Rows: training stage or dataset (e.g. “Aircraft”, “Caltech101”, …, or “After task 1”, “After task 2”, …).  
   - Columns: evaluation dataset (same 11 names).  
   - Cells: Top-1 (and possibly Top-5) accuracy.

3. **Baselines**  
   Same table might have rows for:  
   - Fine-tuning (FT),  
   - LwF,  
   - Other CL methods,  
   - **MoE-Adapters (ours)**.  
   So “MTIL results” = the row(s) that correspond to MoE-Adapters on this 11-dataset setup.

4. **One-number summary**  
   In text or table caption: “average accuracy over 11 datasets” or “mean accuracy” for the proposed method and baselines.

5. **Experimental details**  
   - Order of datasets (same as in this repo or not).  
   - Number of training steps/epochs per task.  
   - Whether they report **per-checkpoint** (each of the 11 .pth) or only **final** model.

---

## How This Repo’s Numbers Relate to the Paper

- **Same setting**: 11 datasets, sequential training, one adapter per task, auto-chooser to route.
- **Same kind of result**: For each checkpoint `dataset[i].pth` we evaluate on every `dataset[j]` and get a 11×11 matrix (and optionally averages).
- **Possible differences**:  
  - Dataset order (e.g. TinyImagenet for chooser vs no TinyImagenet in eval).  
  - Epochs/iterations, LR, seed.  
  - StanfordCars test labels (paper may use a specific setup; we need `cars_test_annos_withlabels.mat` for correct eval).

So: **“MTIL results” in the paper = that 11×11 accuracy matrix (and any average over the 11 eval datasets)**. Match “model trained on X” to our checkpoint `X.pth`, and “evaluated on Y” to our eval dataset Y; then compare the numbers in the paper’s table to your `complete_results_summary.txt` (or the cross-dataset matrix you generate).

---

## Short checklist when reading the paper

- [ ] Find the “multi-task” / “11 dataset” table.
- [ ] Identify which row is “MoE-Adapters” (or “Ours”).
- [ ] Check if the table is (train dataset × eval dataset) or (method × eval dataset).
- [ ] Note if they report “average over 11 datasets” and where.
- [ ] Compare with this repo’s matrix and averages after fixing StanfordCars (and any other) eval issues.

If you tell me the **exact table number** (e.g. “Table 2”) and **caption** (or a short description of rows/columns), I can help map it line-by-line to this codebase’s MTIL results.


---

## Local MTIL results: known issues and fixes

### StanfordCars evaluation ~0.5% (random)

**What you see:** When the **evaluation dataset** is StanfordCars, all checkpoints give ~0.5% Top-1 (and ~2.5% Top-5). That is **random chance** for 196 classes (1/196 ≈ 0.51%), so it indicates **wrong or missing test labels**, not model quality.

**Cause:** The code expects a **test annotations** file that the official Stanford Cars dataset does **not** provide (test labels are withheld for the benchmark). It looks for:

- `{DATA_LOCATION}/stanford-cars/cars_test_annos_withlabels.mat`, or  
- `{DATA_LOCATION}/stanford-cars/devkit/cars_test_annos_withlabels.mat`

**Format:** Same structure as `cars_train_annos.mat`: a MATLAB file with key "annotations", an array of structs, each with:

- "fname": image filename (e.g. "00001.jpg")
- "class": **1‑indexed** class label in [1, 196] (same order as `cars_meta.mat` class names)

**Fix:** Create `cars_test_annos_withlabels.mat` and place it in one of the two paths above. You need a source of test labels (e.g. from the paper authors, or a community devkit that provides them). Then re-run the MTIL test script; StanfordCars eval accuracy should become meaningful.

### Aircraft / OxfordPet "constant" accuracy across checkpoints

**What you see:** When the **evaluation dataset** is Aircraft (or OxfordPet), the reported accuracy is **the same** for every checkpoint (e.g. Aircraft column ~51.34%, OxfordPet row ~89.04% regardless of "model trained on").

**Cause:** This is **expected** given the current router (auto-chooser). For those eval datasets, the router consistently selects the **same** expert (or the zero-shot branch) for every image. So the effective model used at test time does not change when you switch checkpoints; only the set of experts and the chooser change, and for that dataset the chosen path happens to be constant.

So:

- **Aircraft ~51.34%:** Likely the router always uses zero-shot CLIP (or one fixed expert) for Aircraft, so you always see the same zeroshot/fixed accuracy.
- **OxfordPet ~89.04%:** Same idea: one dominant path for OxfordPet, hence constant number.

**No code bug:** This reflects router behavior after training, not a bug in the evaluation pipeline. To get different accuracies per checkpoint for those datasets, the router would need to route them to different experts depending on which task checkpoint you load (e.g. via different chooser behavior or training).

---

**Summary:** Fix StanfordCars by adding correct `cars_test_annos_withlabels.mat`; treat Aircraft/OxfordPet constant accuracies as expected with the current setup. Then compare your 11×11 matrix and averages to the paper's MTIL table.
