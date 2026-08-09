# KLA PS01 — Handoff v2 (supersedes HANDOFF.md §2, §4, §6, §7)

**Date of this handoff:** 9 Aug 2026. **Submission target: 15 Aug** (deadline 16 Aug, portal will be slow).

Sections **§1 (degradation forensics), §3 (architecture), §5 (bugs), §9 (references), §10 (traps)**
of the original HANDOFF.md are still accurate — read them, do not revisit them.
Everything below either supersedes or extends the original.

---

## 0. Status in one line

**A submittable state exists.** Fresh-clone gate is CLOSED. Model trained on a clean,
leakage-free split. All remaining work is writing — no GPU required.

| Item | State |
|---|---|
| `evaluate.py` runs unedited from a fresh clone | ✅ **CLOSED** — md5 `a8f8eb3e7fe278287ae13ac588e607c9` |
| Weights committed at `weights/best_ema.pt` | ✅ 2.35 MB, direct commit, no LFS |
| Clean validation split (no leakage) | ✅ verified by `archive/scripts/verify_split.py` |
| Restored outputs folder in repo | ❌ **TODO** (`gate_out` exists, not committed) |
| `requirements.txt` + `requirements_full.txt` | ❌ **TODO** |
| README | ❌ **TODO** |
| Deck (`TeamName_KLA_PS01.pdf`) | ❌ **TODO** — full spec in §6 below |
| Demo video ≤5 min | ❌ **TODO** |

---

## 1. FINAL NUMBERS — use these everywhere

Model: `RestoreNet`, C=64, B=16, **0.565M params**, EMA checkpoint @ step 14000.
Run: `runs/v1_clean`, 60% KLA / 40% procedural layouts, bs 16, patch 128, lr 1e-3.

| Split | PSNR (dB) | SSIM | LPIPS-alex | What it is |
|---|---|---|---|---|
| `kla` | **27.04** | 0.7710 | 0.2743 | 64 real SEM images, **held out by filename** |
| `layouts_holdout` | 31.94 | 0.9574 | 0.0450 | 32 synthetic layouts, frozen, unseen files |
| `set5_ood` | 29.73 | 0.8650 | 0.1963 | 8 natural photographs (Set5 benchmark) |

**Baselines:** bicubic on `kla` = **22.61 dB / 0.5958**. Headline: **+4.43 dB over bicubic.**
Bicubic on `layouts_holdout` = 24.15 dB / 0.8321. Bicubic on `set5_ood` = 21.66 dB.

**Latency (this is the number KLA benchmarks):**
`evaluate.py`, batched, fp16 AMP, end-to-end (disk read → preprocess → H2D → forward → D2H → disk write),
400 images on a Kaggle T4: **21.0 ms/image at 128→256, and 61.68 ms/image at 256→512**.
Because the hidden test set is roughly half and half, the composite latency is **~41 ms/image**.

**Compute:** 32.5 GMACs (65 GFLOPs) at 256² input.
**Peak VRAM:** 653 MB at bs=8 fp16.

**Derived cost split**:
128→256 (21.0ms) vs 256→512 (61.68ms). Although pixels increased 4×, latency only increased 2.94×.
Solving for fixed vs pixel-dependent cost gives **~7.4 ms fixed per-file overhead, and ~13.6 ms compute** at the 128→256 scale.
Consequence: On an H100, the compute time will collapse to near-zero, meaning **per-file I/O overhead (7.4ms) becomes the dominant cost.** Larger batches / threaded file reads are the only way to break this floor.

---

## 2. What changed since HANDOFF.md — supersedes §2, §4, §6, §7

### 2.1 Validation leakage — FOUND AND FIXED

The `kla` validation split was reading `train/train/GT` — the same directory `kla_packed.npy`
was built from. Every number before this fix was measured partly on training data.
This is HANDOFF §10's "random split instead of held-out content" trap, live.

**Fix:** `pack_data.py --skip 64` drops the first 64 alphabetically-sorted files, which is
exactly the set `train.py`'s val loader selects. `archive/scripts/verify_split.py` asserts
the two sets are identical — **run it before any re-pack.**

**Cost of the leak: 0.04 dB** (27.08 contaminated → 27.04 clean). At 0.565M params there was
never enough capacity to memorise 64 of 3200 images. **This is a slide, not an embarrassment**
— predicting a leak, removing it, and quantifying it is a methodology win.

### 2.2 Corrected data counts (HANDOFF §2's table was stale)

| Source | Actual | HANDOFF said |
|---|---|---|
| `data/kla_packed.npy` | **3136** (3200 minus 64 val) | 2500 |
| `data/layouts_packed.npy` | **4000** | 2000 |
| NFFA-EUROPE | still **not downloaded** | not downloaded |
| DF2K | still **not downloaded** | not downloaded |

⚠️ **`data/layouts/` currently holds 4000 image files** — 2000 originals plus 2000 regenerated
on top without clearing the directory. The packed `.npy` was restored from backup so training
was unaffected, but **any future `pack_data.py --src data/layouts` packs a 50/50 mix of two generations.**
Clean it: `rm -rf data/layouts` then regenerate.

⚠️ **`layout_generator.py` is not reproducible** — regenerating with the same args produced a
different array. Either no seed is plumbed through, or the §5 pitch fix changed output. Fix the
seed before Round 2, since §5 requires regenerating training layouts and a fresh frozen holdout together.

### 2.3 The `layouts_holdout` split is NOT an OOD test

It holds out *files*, not *content* — the same generator produced 40% of the training mix.
It scores 31.94 dB, **higher than the in-distribution `kla` split**, which is diagnostic:
a genuine OOD split always scores worse.

**Never call this "generalization."** The defensible framing, which is still a good story:

> KLA GT is 100% organic SEM. We predicted the hidden OOD half would be patterned/microfabricated
> content, generated procedural layouts to cover that prediction, trained on them, and used a
> frozen holdout to verify the coverage held on unseen instances.

That is a *bet placed and validated*. A juror who asks "what did you train on?" collapses the
generalization claim instantly; the coverage claim survives.

`set5_ood` (8 natural photographs) is the only true cross-domain split, and it's weak evidence —
tiny, and the wrong domain entirely. **NFFA-EUROPE is still the right OOD split and is still not downloaded.**

### 2.4 U-Net alternative — TESTED AND REJECTED (this is your best slide)

A NAFNet-style U-Net was trained and compared. It won LPIPS-alex on `kla` by 0.108 (0.1544 vs 0.2624).
**Visual inspection reversed the ranking.**

The crop grid (`hallucination_check.png`, 5 rows × 4 columns: Bicubic / GT / v1 / U-Net) shows:

- The U-Net's "detail" inside contacts is **the same grain pattern present in the bicubic input** —
  it is under-denoising and passing speckle through. LPIPS-alex rewards high-frequency energy
  regardless of whether it corresponds to anything real.
- **Row 3:** GT shows a smooth bright particle (a defect). The U-Net invents internal dark structure
  that does not exist — a phantom sub-feature on the exact object an inspection tool cares about.
- **Row 4:** GT dots have dark centres (rings). The U-Net fills them in, destroying real structure
  while its LPIPS improves.

**Backbone check:** LPIPS-alex gap 0.1080 → LPIPS-vgg gap 0.0349. **~68% of the U-Net's advantage
was AlexNet-specific**, i.e. metric-gaming rather than image quality.

**Honest caveat for the slide:** the U-Net still wins in-distribution LPIPS under *both* backbones.
Say "we accepted an LPIPS penalty in exchange for fidelity," not "its advantage was an illusion."
The first survives questioning; the second doesn't.

Full U-Net numbers (measured on the contaminated split — comparison is still valid, both models
were scored identically):

| Split | PSNR | SSIM | LPIPS-alex | LPIPS-vgg |
|---|---|---|---|---|
| kla | 26.31 | 0.7461 | 0.1544 | 0.3441 |
| layouts_holdout | 27.02 | 0.8849 | 0.1552 | 0.2220 |
| set5_ood | 28.74 | 0.8420 | 0.1065 | 0.2777 |

### 2.5 Capacity is NOT exhausted — key Round 2 lead

HANDOFF §6 predicted 28.5–29.5 dB at 20–40k. **Actual: flat at 27.0 by 12k.** The prediction is falsified.

But a 96-channel / 20-block variant reached **26.98 dB on one fifth the sample budget**
(8000 steps × bs 8 = 64k samples, vs 20000 × 16 = 320k). A 2.8×-FLOPs model landing within
0.06 dB on 20% of the data is evidence *against* a capacity ceiling.

[Guessing] the real ceiling is **information-theoretic** — speckle at σ up to 0.315 plus 2×
downsampling destroys detail no model recovers. **Test in Round 2:** bin validation PSNR by σ.
Flat across σ ⇒ operator-limited. Steep at low σ ⇒ capacity-limited. This single experiment
decides the entire Round 2 scaling strategy.

### 2.6 RAW vs EMA — ship EMA

RAW LPIPS on `kla` across steps 10k/11k/12k: 0.2702 → 0.2602 → 0.2780. **Step-to-step noise ≈ ±0.02.**
EMA over the same steps: 0.2757 → 0.2755 → 0.2760. **Noise ≈ ±0.0005.**

RAW occasionally scores better, but which checkpoint gets the good sample is luck, and the
advantage won't transfer to a different test set. **Ship EMA — its number is real.**

**General rule, learned the hard way:** on 64 validation images, differences under **0.02 LPIPS**
or **0.1 dB PSNR** between adjacent checkpoints are noise. Do not act on them. This invalidated
two separate "trends" during development.

### 2.7 `best_ema.pt` selection criterion is flawed (known, not fixed)

`train.py` saves on improvement in **mean PSNR across all three splits**. Two problems:
it ignores SSIM and LPIPS entirely, and it weights `set5_ood` (8 irrelevant photographs) equally
with `kla` (the split that proxies the real test set). Late-run "improvements" were carried almost
entirely by `layouts_holdout`. Fix in Round 2: select on `kla` LPIPS or a weighted composite.

### 2.8 `evaluate.py` hardening (all shipped)

- **Checkpoint path resolved relative to `__file__`**, never CWD. A judge runs from anywhere.
- **Key mismatch is now FATAL** with a diagnostic listing missing/unexpected keys and the built config.
  Previously `strict=False` swallowed every key and emitted pure bicubic while printing success.
- **Pre-flight self-test:** runs a random tensor, compares against bicubic, aborts if the delta is
  < 1e-6. Works because the zero-init tail makes an unloaded model *bitwise* bicubic. Expected
  healthy value ≈ **0.30–0.38**. This caught a real stale checkpoint.
- **Reflect padding** for arbitrary input shapes. Verified on 250×250, 257×301, 15×17.
- ⚠️ **A `step` assertion was added and then removed** — `train.py` never writes that key, so it
  fired on every valid checkpoint. Do not re-add it without also writing `sd["step"] = step` at save time.
- **Per-image bicubic fallback stays non-fatal** (better 399 good + 1 bicubic than a crash), but
  it must print `n_fallback/n_total` loudly at the end.

### 2.9 New code artifacts

| File | Purpose |
|---|---|
| `verify_outputs.py` | Post-inference gate: proves output ≠ bicubic, filenames/dtypes/shapes round-trip. **Run after every `evaluate.py` run.** Healthy output-vs-bicubic PSNR ≈ **24 dB**. |
| `archive/scripts/verify_split.py` | Asserts `pack_data.py --skip N` matches the val loader's file set |
| `archive/scripts/compare_models.py` | Multi-model, multi-split, multi-metric comparison |
| `lpips_metric.py` | LPIPS for validation + optional loss |
| `pack_data.py --skip N` | Excludes the first N sorted files from packing |

**`lpips_metric.py` design constraints — do not break these:**
1. Backbone cached in a **module-level dict**, never a registered submodule. If it were a child
   module it would land in `model.state_dict()` and inflate `best_ema.pt` from 2.35 MB to ~10 MB
   (alex) or ~60 MB (vgg). The commit-weights-directly plan depends on this.
2. Runs **fp32 with autocast explicitly disabled** — a fp16 backbone is an `inf` source, and
   HANDOFF §5.2 already documents a GradScaler/inf deadlock.
3. Metric uses `clamp=True` (mirrors what `evaluate.py` writes to disk); loss uses `clamp=False`
   (clamping kills gradients on saturated pixels).
4. 🔴 **`evaluate.py` must NEVER import `lpips`, directly or transitively.** It triggers a 233 MB
   AlexNet download at runtime. On a restricted-egress benchmark box that is a hard stop.
   `lpips` goes in `requirements-train.txt` only.

### 2.10 Operational notes (Kaggle)

- **2× Tesla T4** available, not 1. DDP is viable for the Round 2 300k run.
- IPython `!` **refuses a trailing `&`** (`OSError: Background processes not supported`).
  Use `subprocess.Popen(..., start_new_session=True)` — this also survives a kernel restart.
- Chain shell steps in **one** cell with `&&` or use Python with `check=True`. Separate `!` lines
  do not stop on failure, which once nearly launched a 4-hour run on the wrong packed data.
- Always `CUDA_VISIBLE_DEVICES=0` on launch, and `pgrep -af train.py` before launching.
  Two concurrent runs on one T4 caused an OOM (8.83 GiB + 5.72 GiB of 14.56 GiB).
- `--out` must be an **absolute path under `/kaggle/working`** or it won't persist.
- ⚠️ **`weights/best_ema.pt` is overwritten every `--save_every` steps while training runs.**
  Copy it to the repo *after* the run ends, or the committed file won't match the reported numbers.
- ⚠️ **Repo clones at 53 MiB** for a 2.35 MB model — ~50 MB of history (old checkpoints / packed
  arrays / repeatedly-committed `submission.zip`). Not disqualifying. Do **not** rewrite history
  this close to the deadline; just stop committing generated artifacts.

---

## 3. Submission requirements — the complete checklist

### 3.1 Deliverables

| # | Item | Rule |
|---|---|---|
| 1 | `TeamName_KLA_PS01.pdf` | **8–9 slides**, instruction slide deleted. Wrong filename or >9 slides = zero. |
| 2 | Public GitHub repo | Must contain items 3–7 below |
| 3 | `README.md` | Clone → restored images with **no contact required** |
| 4 | `evaluate.py` | **Standalone script, NOT a notebook.** Takes `--input_directory` / `--output_directory`. Must run **AS-IS** on KLA's H100. |
| 5 | Training script | `train.py` — judges read it for reproducible seeds, deterministic ops, logging |
| 6 | Model weights | In-repo at `weights/best_ema.pt`. **No Drive links** — a dead link is a zero. |
| 7 | Restored test outputs | Model outputs on your own held-out split (no official test set at Round 1). **Label the split clearly.** |
| 8 | `requirements.txt` | Minimal: `torch`, `numpy`, `opencv-python`. Plus `requirements_full.txt` (literal `pip freeze`) — the rules ask for a freeze, but a raw freeze pins `nvidia-*` wheels that break elsewhere. Ship both, explain which to install. |
| 9 | Demo video | Optional, ≤5 min |

### 3.2 Citation requirement (explicit at the webinar)

Any external data or pretrained weights **must** be cited in the deck AND README with
names, links, licenses, and papers. Currently in use: **Set5** (`set5_ood` split).
If NFFA-EUROPE is added: CC-BY, attribution required, cite Aversa et al. 2018.

### 3.3 Scoring pillars (weighting is hidden — do not assume one dominates)

1. **Restoration quality & generalization** — PSNR, SSIM, LPIPS, on in-distribution *and* OOD halves.
   No clipping or normalization applied by judges; your direct output is scored.
2. **End-to-end inference speed** — the *whole* pipeline on an H100, not just the forward pass.
   Batch processing explicitly encouraged.
3. **Training hygiene & reproducibility** — code must run out of the box.
   **This is a full third of the rubric and is entirely writing.**

### 3.4 Timeline

| Date | Event |
|---|---|
| **15 Aug** | **Submit (target)** |
| 16 Aug | Deadline — portal will be slow, do not wait |
| 17–26 Aug | Round 1 evaluation |
| 27 Aug | Top 30 announced |
| 28 Aug – 4 Sep | **Round 2 — the H100 benchmark actually runs here** |
| 6 Sep | Top 10 |
| 17–18 Sep | Grand finale, Yashobhoomi, New Delhi |

---

## 4. Remaining work, in priority order

**All of this is writing. None needs a GPU.**

1. **`requirements.txt` + `requirements_full.txt`** (30 min).
   First: `grep -n "^import\|^from" evaluate.py model.py | grep -i "lpips\|torchvision"` — must be empty.
2. **README** (1 hr). Clone → install → run. Cite Set5. State which requirements file to install.
   Note honestly which §7.1 gate conditions were verified (a "different person" ran it, or not).
3. **Restored outputs** (15 min). Commit `gate_out/`, labelled as your own held-out split.
4. **Measure bicubic on `set5_ood`** (5 min). S7 has a hole without it.
5. **Deck** (see §6). The largest remaining item.
6. **Video** (2 hrs). Screen-record clone → install → run → crop comparisons.

**Optional, only if 1–6 are done by Wednesday:** LPIPS training-loss experiment.
`w_lp = 0.10` after a 5k warm-up, `net="alex"`. [Guess] 0.03–0.06 LPIPS for 0.1–0.3 dB PSNR.
🔴 **Mandatory gate:** train 2k steps, regenerate the hallucination crop grid, and if grain
reappears inside the contacts, **kill the term.** You have direct evidence that optimizing
LPIPS-alex on this task rewards retained speckle. Also validate on `vgg` — if vgg doesn't move,
you're gaming the metric.

---

## 5. Round 2 plan (if top 30) — 28 Aug – 4 Sep

Highest-value first:

1. **The σ-binning experiment** (§2.5). Decides whether to scale capacity or stop. Do this first.
2. **Download NFFA-EUROPE + DF2K**, dedupe against KLA GT, run the full 4-source mix.
   This gives you a real OOD split for the first time.
3. **Harden `layout_generator.py`** per HANDOFF §5: near-Nyquist pitches, denser `logic` polygons,
   overlaid second layer at ~60% opacity with a different pitch, defect features (particles, bridges,
   opens). Target: layouts bicubic **at or below** the KLA baseline of 22.61 dB. Fix the seed.
   Regenerate training layouts **and** a fresh frozen holdout together.
4. **96/20 at matched sample budget** — 20k steps at bs 16. §2.5 says this is likely to win.
5. **Fix checkpoint selection** to use `kla` LPIPS or a weighted composite (§2.7).
6. **INT8/FP8 quantization study** + FLOPs/params/latency table. Aimed at the KLA juror.
7. **I/O optimization** — 56% of end-to-end time. Async prefetch, larger batched reads.
   Higher leverage than any architecture change.
8. TTA cost/benefit (×8 self-ensemble ≈ +0.15 dB for 8× latency — probably not worth it).
9. Large teacher → small student distillation. 300k-step run.

**Jury targeting:** KLA juror **Akshat Singh** — vision foundation models, self-supervised learning,
model compression, quantization, efficient GPU inference. Applied Materials juror: **Aayush Raina**.

**Finale:** lead with deployment (images/wafer/hour, integration into the inspection pipeline,
false-detail risk analysis), **not** architecture. Laptop demo + recorded video fallback.

---

## 6. THE DECK — complete specification

**File:** `TeamName_KLA_PS01.pdf`. **Exactly 9 slides.** Instruction slide deleted.
Replace `TeamName` with the actual registered team name.

### Narrative spine
Three things make this submission distinctive, and none of them is the PSNR:
1. **We measured the degradation operator instead of trusting the problem statement — and the
   problem statement was wrong.**
2. **We caught LPIPS ranking a worse model first, and reversed it on visual inspection.**
3. **We found and quantified our own validation leakage.**

All three are judgment stories. Judgment is rarer than a 27 dB PSNR and it is what a metrology
company is hiring for. Lead with them.

---

**S1 — Title**
Team name, members, PS01 "AI-Based Restoration of Degraded Images for Semiconductor Inspection", KLA.
One-line result: *0.565M params · +4.43 dB over bicubic · 21 ms/image end-to-end.*

**S2 — Problem & approach**
The task (joint denoise + ×2 SR, grayscale, half-OOD test set, speed scored).
The thesis in one sentence: **"We measured the degradation rather than assuming it, and built the
data engine around the measurement."** Pipeline diagram: measured degradation → data mixture →
RestoreNet → clamped output.
*Asset needed: pipeline diagram.*

**S3 — Degradation forensics** ← strongest technical slide
Measured over 200 training pairs:
- Downsample operator = `cv2.INTER_CUBIC`, **no antialiasing** (antialiased variants 3.3% worse)
- Sub-pixel shift 0.000 ± 0.09 px — none
- Mean preservation 1.00008 ⇒ noise is **exactly zero-mean multiplicative**
- `var = a·μ² + b·μ + c`: `a` well-determined, `b` and `c` straddle zero ⇒ **no Poisson, no additive Gaussian**
- **Gamma (Goodman L-look) speckle.** Measured skew +0.324 vs 2/√L = 0.343 predicted at L = 1/σ² = 34.1.
  Confirmed by an unfitted moment.
- σ median 0.153 (p10 0.065, p90 0.221)

**The punchline:** the problem statement lists "Gaussian Noise — image appears soft and hazy" as a
separate degradation. That describes blur. **There is no meaningful Gaussian noise and no separate
blur.** State this plainly — most teams will have implemented the prose.

Consequences: PSNR uses `data_range=1.0` and GT provably cannot leave [0,1], so `clamp(0,1)` at
inference is free score (but never during training — it kills gradients on saturated pixels).
Do **not** min-max normalise the input; it legitimately reaches 1.36 and rescaling destroys
intensity calibration.
*Asset needed: the std-vs-intensity forensics plot.*

**S4 — Data engine & OOD-by-construction**
Mixture: 60% KLA GT (3136) / 40% procedural layouts (4000).
The bet: KLA GT is **100% organic SEM, 0% patterned** (verified over 100 random images).
The hidden OOD half is almost certainly microfabricated content — so generate it:
procedural DRAM / FinFET / contacts / logic / grating SEM.
Three validation splits, each with a stated role. **Label `layouts_holdout` as coverage
verification, not generalization** (§2.3).
Mention the NFFA-EUROPE hypothesis as a stated, unresolved lead — showing you know where the
data probably comes from is worth a line even unconfirmed.
Cite Set5. Cite NFFA (CC-BY) if used.

**S5 — Architecture & the no-hallucination rationale**
`RestoreNet`: 0.565M params, C=64, 16 NAFBlocks, body at **input** resolution, PixelShuffle(2) tail,
global bicubic residual with zero-init tail (so step 0 is bitwise bicubic — doubles as a pipeline oracle).
5×5 depthwise conv, GroupNorm(1,·), SimpleGate, simplified channel attention.

**Plain residual stack, not a U-Net — deliberate.** No downsampling anywhere ⇒ any input size works
with no tiling logic ⇒ `evaluate.py` robustness. Verified on 250×250, 257×301, 15×17.

**Rejected: diffusion and GAN priors.** In metrology a plausible-but-invented pixel is a phantom
defect. Frame as a design decision, not an omission. Pre-empt the obvious follow-up: if the LPIPS
loss term ships, say explicitly it is a *regularizer on a pixel-fidelity objective*, not a
generative prior — no adversarial term, no stochastic sampling, deterministic output.

Loss: `1.0·Charbonnier + 0.2·(1−SSIM) + 0.05·FFT-L1 + 0.05·gradient`.

**S6 — Key finding: LPIPS ranked the wrong model first** ← the slide that wins the round
The U-Net story from §2.4, with the crop grid as the centrepiece. Point at Row 3 (invented internal
structure in a defect particle) and Row 4 (filled-in ring centres, real structure destroyed).
State the backbone result: alex gap 0.1080 → vgg gap 0.0349, **68% metric-specific**.
Close with the honest version: *we accepted an LPIPS penalty in exchange for fidelity, because in
inspection a false detail is worse than a soft edge.*
*Asset: `hallucination_check.png`, cropped to Rows 3 and 4 for legibility.*

**S7 — Results**
Three splits × three metrics, against bicubic (and BM3D+bicubic / EDSR-baseline if you have time
to run them). Headline **+4.43 dB on `kla`**.
Before/after as **zoomed 128 px crops**, never full images.
**One honest failure case** — v1's real weakness is mild over-smoothing: dot rims slightly
lower-contrast than GT. Show it.
Include the leakage note: found it, removed it, cost 0.04 dB.
*Asset needed: bicubic baseline on `set5_ood` (not yet measured).*

**S8 — Efficiency**
Params 0.565M · GFLOPs · **21.0 ms/image end-to-end (T4, batched fp16)** at 128→256 ·
and the 256→512 figure (~4×, measured scaling 3.98×) · peak VRAM.
**The sophisticated point:** ~56% of end-to-end time is disk I/O, not compute, so H100 gains are
bounded by storage rather than FLOPs. Say it — it shows you profiled the pipeline the webinar
described rather than just the forward pass.
Note quantization-readiness (INT8/FP8) and reparameterisation as Round 2 work.
State the measurement boundary explicitly: disk → preprocess → H2D → forward → D2H → disk.

**S9 — References** (from HANDOFF §9, all still correct)
Chen et al. NAFNet ECCV 2022 · Aversa et al. NFFA-EUROPE Sci. Data 5:180172 (2018), CC-BY ·
Goodman speckle / L-look Gamma · Jiang et al. Focal Frequency Loss ICCV 2021 ·
Chen et al. Masked Image Training CVPR 2023 · Wang et al. Real-ESRGAN ICCVW 2021 ·
Zhang et al. BSRGAN ICCV 2021 · Ren et al. NTIRE 2026 Efficient SR Report CVPRW 2026 ·
Zhang et al. LPIPS CVPR 2018 · Set5 · Oxford *Microscopy and Microanalysis* 2025 SEM denoising
survey · Nature Sci. Rep. 2025 (Restormer/NAFNet/HINet/CGNet on charging-condition SEM).

### Assets to prepare before deck-building

| Asset | Status |
|---|---|
| `hallucination_check.png` (5×4 crop grid) | ✅ have — crop to Rows 3–4 for S6 |
| std-vs-intensity forensics plot | ❓ from `degradation_forensics.py` |
| Pipeline diagram | ❌ make |
| Before/after 128 px crops + failure case | ❌ make |
| Bicubic baseline on `set5_ood` | ✅ 21.66 dB |
| GFLOPs + peak VRAM | ✅ 32.5 GMACs (65 GFLOPs) / 653 MB |
| 256→512 latency through `evaluate.py` | ✅ 61.68 ms/image |

---

## 7. Traps that still zero a submission

- Wrong filename or >9 slides
- `evaluate.py` needing an edit, or `--channels`/`--blocks` mismatching the checkpoint
  *(mitigated: architecture should be inferred from the checkpoint, not CLI defaults —
  `C = sd["head.weight"].shape[0]`, `B = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))`)*
- Weights behind a dead Drive link
- Renamed or bit-depth-mangled outputs — **matching is by filename**
- Random 90/10 validation split instead of held-out content *(fixed — keep it fixed)*
- Trusting the problem statement's noise description over measurement
- **`lpips` or `torchvision` in `evaluate.py`'s import chain** — 233 MB runtime download
- Submitting on 16 Aug — target the 15th
