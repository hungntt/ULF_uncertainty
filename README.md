# Lesion Preservation Is Invisible to Image-Quality Metrics in Ultra-Low-Field MRI Enhancement

Official code for the paper of the same name (`manuscript/main.tex`). We study 64 mT (Hyperfine) → 3T MRI enhancement
on 50 paired POCEMR subjects and show that PSNR/SSIM and voxel-level conformal guarantees do not reflect whether
lesions survive enhancement, using simulated lesions and a simulation-trained detectability map.

The code covers paired baselines, learning the forward (3T → 64 mT) operator to exploit an unpaired 3T corpus,
cross-contrast (T1/T2) fusion, and uncertainty / QA / lesion-detectability maps with conformal guarantees.

## Layout

| path | contents |
|---|---|
| `ulfkit/` | shared library: `data.py` (paired cohort, subject-level splits), `models.py` (residual U-Net, LoHiResGAN, PatchGAN), `train.py` (training loop), `metrics.py` (whole-image and brain-masked PSNR/SSIM, per-sub-band cosine, paired tests), `eco.py` (ECO-MT components) |
| `exp/` | experiment scripts (`*.py`) and the shell drivers that chain them (`run_*.sh`, `chain*.sh`) |
| `ideas/FINDINGS.md` | running log of results and caveats |
| `manuscript/`, `paper/` | LaTeX sources (IEEE ISBI template) and figure scripts |
| `fig/`, `reports/` | generated figures and HTML reports |
| `out/` | checkpoints, JSON metrics, logs (git-ignored) |
| `datasets/` | raw archives and extracted data (git-ignored) |

## Experiments (`exp/`)

| script | question |
|---|---|
| `baselines.py` | identity, U-Net, pix2pix, LoHiResGAN, and a subject-mismatched-pairs control |
| `ceiling.py` | cross-subject vs within-subject residual cosine per wavelet sub-band |
| `d1.py`, `d1_*.py` | Direction 1: learn the 3T → 64 mT operator, spectrally calibrate the 3T corpus, pretrain on pseudo-pairs (`forward` / `pairs` / `train`) |
| `d2.py` | Direction 2: availability-conditioned T1/T2 fusion with modality dropout |
| `d3.py` | Direction 3: CQR / Gaussian-NLL intervals, split-conformal and conformal risk control, calibration-subject sweep |
| `d3qa*.py` | cross-fitted QA (error) map, evaluation and deployment |
| `d3lesion.py`, `d3finding.py`, `d3detect*.py`, `d3ood.py` | simulated-lesion retention, finding-aware risk control, detectability map |
| `d3real*.py` | screen the cohort for real focal lesions |
| `d3acq.py`, `d3cascade*.py`, `d3tissue.py`, `d3arch*.py` | re-acquisition guidance, error reducibility, tissue-segmentation proxy, architecture comparison |
| `ecomt*.py` | ECO-MT: gated Mean Teacher on unpaired ULF |
| `d3zoo.py`, `d3zoo_stats.py`, `run_seeds12k*.sh`, `run_zoo_rolling.sh` | 15 enhancers / 39 models (six architectures x five seeds at 12000 steps): retention, SSIM/PSNR, tissue proxy, exported volumes; Spearman rho with a bootstrap over enhancers, seeds and subjects |
| `d3synthseg.py` | SynthSeg (FreeSurfer 8.2) Dice and volume error against SynthSeg on 3T, as the downstream endpoint |
| `d3lesiontype.py`, `d3polarity.py`, `d3pipe.py` | cross-contrast routes into 3T T1: wrong-sign rate by T1/T2 polarity quadrant |
| `d3synthsr.py` | the same quadrants through the public SynthSR tools (`--lowfield`, Hyperfine), 3D |
| `d3realcoupling.py` | real-anatomy check of the inversion (negative: the healthy cohort has no strong coupling-violating structure) |
| `report.py`, `figs.py`, `qualitative.py` | result tables and figures |

Each script's docstring states the question it tests. The `run_*.sh` / `chain*.sh` files record the exact commands and
step counts used for each result.

## Running

```bash
# pytorch, pytorch_wavelets, numpy, scipy, scikit-learn, matplotlib; CUDA GPU required
export ULF_DATA=/path/to/preprocessed/npy      # default: datasets/npy (.npy slabs + manifests)
python -I -u exp/baselines.py --contrast T2 --method unet --steps 12000
python -I -u exp/d1.py forward --contrast T2 --steps 6000
python -I -u exp/d1.py pairs   --contrast T2 --mode learned_cal
python -I -u exp/d1.py train   --contrast T2 --mode learned_cal
python -I -u exp/d2.py --use T1,T2 --out both --steps 10000 --tag joint --drop
python -I -u exp/d3.py ...                     # see run_all.sh for full arguments
python exp/report.py                           # tables from out/*.json
```

## Data and evaluation protocol

- 40 paired training subjects (32 train / 8 dev-val) and 10 locked test subjects; 1 mm, 256×256, [0,1], 2.5D slabs.
  Unpaired pools: FOMO-50k / IXI 3T and 48 ULF-only subjects.
- Volumes are skull-stripped (~75% of each slice is zero), so metrics are reported **brain-masked** as well as whole-image.
- All comparisons are subject-level (Wilcoxon, bootstrap) on the locked test set.

## Known gaps

- Scripts hardcode `/mnt/G/Projects/ULF-MICCAI` (and some `CACHE` paths and a venv in the `run_*.sh` files); edit these for your machine.
- The preprocessing step that turns `datasets/raw/*.7z` into the `.npy` arrays and manifests is not in this repo.
- Key results and caveats are in `ideas/FINDINGS.md`.
