"""Filesystem layout and dataset constants.

Small inputs (the subject x ROI tables, the phenotypic CSVs, the Yeo atlas)
live inside the repository under ``data/``.  The raw per-subject ``.mat``
archives are large and stay outside it; point :data:`ROIS_DIRS` at wherever
they live, or set the ``IBB_RAW_DIR`` environment variable.
"""

import os
from pathlib import Path

#: Repository root, resolved from this file's location.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
RESULTS_DIR = ROOT / "results"

#: Where the untracked per-subject ``.mat`` archives live.
#: Override with ``export IBB_RAW_DIR=/path/to/adhd200``.
RAW_MAT_ROOT = Path(os.environ.get("IBB_RAW_DIR", ROOT.parent))

#: Per-site directories of ROI time series, relative to :data:`RAW_MAT_ROOT`.
#: Note the leading space in the inner directory name — it is in the ADHD-200
#: distribution as shipped, not a typo here.
ROIS_DIRS = {
    "peking": RAW_MAT_ROOT / "rois_1000_beijing" / " rois",
    "nyu": RAW_MAT_ROOT / "rois_1000_nyu" / "rois",
}

# --- tracked inputs -------------------------------------------------------
CASE_TABLE = PROCESSED_DIR / "pku_caseGroup.tsv"
CONTROL_TABLE = PROCESSED_DIR / "pku_controlGroup.tsv"

PHENOTYPIC = {
    "peking": RAW_DIR / "Peking_phenotypic.csv",
    "nyu": RAW_DIR / "NYU_phenotypic.csv",
}

YEO7_ATLAS = EXTERNAL_DIR / "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"
YEO7_ATLAS_LIBERAL = (
    EXTERNAL_DIR / "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz"
)

#: ROI parcellation matching the 954 columns of the subject tables.
BRAIN_ROIS = EXTERNAL_DIR / "brain_rois.nii.gz"
ROI_NETWORK_MAPPING = PROCESSED_DIR / "roi_to_network_mapping.csv"

#: Cortical-thickness groups for the second application: gifted (GG) vs
#: control (CG), 15 and 14 subjects over 308 vertices. Both are d > n.
RAW_CT = {
    "control": RAW_DIR / "rawCT_CG.mat",
    "gifted": RAW_DIR / "rawCT_GG.mat",
}
#: Variable name inside both ``.mat`` files.
RAW_CT_KEY = "rawCT"

# --- dataset constants ----------------------------------------------------
#: Number of ROIs in the ROI-1000 parcellation as distributed (954 survive).
N_ROIS = 954

#: Number of Yeo networks, excluding background.
N_NETWORKS = 7

#: The subject features are time-averaged z-scores and land around 1e-17, which
#: is small enough that the HMC step sizes would have to change by orders of
#: magnitude.  The analysis multiplies them up to O(1) first.  It is a pure
#: rescaling of the data: correlation estimates are unaffected, covariance
#: estimates scale by ``FEATURE_SCALE**2``.
FEATURE_SCALE = 1e17

#: Control subjects are subsampled to match the 78 cases.  Fixed for
#: reproducibility — the original notebook called ``.sample(n=78)`` unseeded.
N_CONTROL_MATCHED = 78
CONTROL_SAMPLE_SEED = 0


def ensure_dirs():
    """Create the output directories if they do not exist."""
    for d in (PROCESSED_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
