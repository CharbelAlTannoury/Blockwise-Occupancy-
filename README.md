# Blockwise occupancy inference under weak dependence

Technical appendix and Python code for *Blockwise Occupancy Inference under Weak Dependence with Record-Based Persistence Thresholds*, by Charbel Al Tannoury, Anis Hoayek, Youssef Salman, Hyam Abboud, and Mireille Batton-Hubert.

## Contents

- [Technical appendix](appendix.pdf): proofs, additional comparisons, extended simulation analyses in Appendix F, and complete CWRU application results in Appendix G.
- appendix_source/: the source files needed to compile the appendix.
- simulation_study.py: fitted CHARN experiments, count-law calculations, and diagnostic-corruption experiments.
- calibration_sensitivity.py: reference-length and gap experiments, fitted rank comparisons, and numerical mixing bounds.
- count_distributions.py: distribution utilities used by the simulation programs.
- cwru/cwru_analysis.py: healthy-reference fitting, calibration, monitoring, aggregation comparisons, and sensitivity analyses for the 56 CWRU recordings.
- cwru/verify_results.py: independent numerical checks and reconciliation of the CWRU block-level results.
- cwru/make_outputs.py: reproduction of the CWRU tables and figures.
- cwru/data/data_manifest.json: original data URLs, exact channel names, experimental conditions, checksums, and acquisition-quality notes.
- cwru/results/: full-precision results, frozen fits, and block-level classifications.
- requirements.txt: the tested Python package versions.

## Reproducing the simulations

The programs were tested with Python 3.12. Install the dependencies and run from the repository root:

~~~sh
python3 -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 simulation_study.py --replications 1000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 calibration_sensitivity.py --replications 1000
~~~

The simulation seeds are 20260919 and 20261107. These commands generate numerical tables, figures, and experiment metadata. The simulation studies do not require an external dataset.

## Reproducing the CWRU application

~~~sh
python3 cwru/cwru_analysis.py
python3 cwru/verify_results.py
python3 cwru/make_outputs.py
~~~

The first command downloads missing original MATLAB files directly from the CWRU Bearing Data Center, verifies their SHA-256 digests, and evaluates all nine designs. Python reads the arrays through SciPy. The source downloads occupy approximately 400 MB and are not included in the repository. Existing data are reused after checksum verification.

The source manifest selects X099_DE_time from file 99, which also contains X098 arrays, and X173_DE_time from official file 174. These are deliberate choices matching the published records. The master seed for randomized rank comparisons is 20261110.

The primary specification uses a separate healthy reference for each load, 2,000 fitting transitions, 200 point-calibration scores, 199 reference blocks, micro-window length and spacing 10, and 20 micro-scores per monitored block. All preprocessing estimates and thresholds remain frozen during evaluation. The record floor is 6.

The metrics CSV includes all nine designs. The compressed primary_blocks.csv.gz contains all 56 recordings evaluated under the primary design, including the 16 additional outer-race recordings. Fields ending in rate are fractions; percentages in the paper multiply them by 100. Reported cohort averages weight each recording equally. They are not pooled proportions across recordings of unequal length.

Raw samples are ordered chronologically. Monitoring blocks overlap, and their number is not a number of independent experiments. The known bearing conditions do not annotate individual episode durations. The analysis therefore reports diagnostic occupancy, without assigning physical duration coverage or a numerically certified mixing allowance.

The CWRU program generates model files and CSV results under cwru/results/. The output program generates vector figures and LaTeX tables under cwru/publication/. The appendix source contains the published numerical tables at full documented precision.

## Compiling and reading the appendix

With a standard TeX Live installation:

~~~sh
cd appendix_source
latexmk -pdf appendix.tex
~~~

Keep the compiled appendix.pdf beside the manuscript named main.pdf when reading downloaded copies. In a PDF reader supporting links between documents, the numbered links lead to the referenced theorem, proof, or appendix section. GitHub's PDF preview does not provide that behavior.

## Data sources

- [CWRU normal baseline recordings](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)
- [CWRU 48 kHz drive-end fault recordings](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)
- [CWRU apparatus and procedures](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures)

Smith, W. A. and Randall, R. B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. *Mechanical Systems and Signal Processing*, 64–65, 100–131. https://doi.org/10.1016/j.ymssp.2015.04.021

The original data remain attributable to CWRU. This repository does not assign a new license to those data.
