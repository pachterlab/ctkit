# verify links by visiting TCIA - manifest URL is what I get by clicking big blue "download" button for imaging data, and metadata url is what I get by clicking small "view" link under Metadata column
tcia_dataset_to_info = {
    "tcga-kirc": {
        "project": "tcga",
        "cancer_organ": "kidney",
        "cancer_type": "clear cell renal clear cell carcinoma",
        "ONCOTREE_CODE": "KIDNEY",
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
        "xdim,ydim,zdim_masked": (185, 185, 75),  # dimensions to standardize to when do_masking is True - set to (None, None, None) to use 95th percentile of extents across all series
        "xdim,ydim,zdim_unmasked": (625, 625, 200)  # dimensions to standardize to when do_masking is False - set to (None, None, None) to use 95th percentile of extents across all series
    },
    "tcga-lihc": {
        "project": "tcga",
        "cancer_organ": "liver",
        "cancer_type": "hepatocellular carcinoma",
        "ONCOTREE_CODE": "LIVER",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-LIHC-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["liver"],
        "clip_min,clip_max": (-200, 400),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-blca": {
        "project": "tcga",
        "cancer_organ": "bladder",
        "cancer_type": "urothelial carcinoma",
        "ONCOTREE_CODE": "BLADDER",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCGA-BLCA-August-30-2019-NBIA-manifest.tcia",  # None for default
        "metadata_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCGA-BLCA-August-30-2019-NBIA-manifes-nbia-digest.xlsx",  # None for default
        "totalsegmentator_organs": ["urinary_bladder"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    # "tcga-brca": {
    #     "project": "tcga",
    #     "cancer_organ": "breast",
    #     "cancer_type": "breast invasive carcinoma",
    #     "ONCOTREE_CODE": "BREAST",
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["breasts"],
    #     "totalsegmentator_task": "breasts",  # None/omit for total
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    # "tcga-cesc": {
    #     "project": "tcga",
    #     "cancer_organ": "cervix",
    #     "cancer_type": "cervical squamous cell carcinoma",
    #     "ONCOTREE_CODE": "CERVIX",
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": [],
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    "tcga-coad": {
        "project": "tcga",
        "cancer_organ": "colon",
        "cancer_type": "colon adenocarcinoma",
        "ONCOTREE_CODE": "BOWEL",
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["colon"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-esca": {
        "project": "tcga",
        "cancer_organ": "esophagus",
        "cancer_type": "esophageal carcinoma",
        "ONCOTREE_CODE": "ESOPHAGUS",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-ESCA-09-16-2015.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["esophagus"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    # "tcga-gbm": {
    #     "project": "tcga",
    #     "cancer_organ": "brain",
    #     "cancer_type": "glioblastoma multiforme",
    #     "ONCOTREE_CODE": "BRAIN",
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["brain"],
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    # "tcga-hnsc": {
    #     "project": "tcga",
    #     "cancer_organ": "head_neck",
    #     "cancer_type": "head and neck squamous cell carcinoma",
    #     "ONCOTREE_CODE": "HEAD_NECK",
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["skull"],
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    "tcga-kich": {
        "project": "tcga",
        "cancer_organ": "kidney",
        "cancer_type": "chromophobe renal cell carcinoma",
        "ONCOTREE_CODE": "KIDNEY",
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-kirp": {
        "project": "tcga",
        "cancer_organ": "kidney",
        "cancer_type": "papillary renal cell carcinoma",
        "ONCOTREE_CODE": "KIDNEY",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-KIRP-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    # "tcga-lgg": {
    #     "project": "tcga",
    #     "cancer_organ": "brain",
    #     "cancer_type": "lower grade glioma",
    #     "ONCOTREE_CODE": "BRAIN",
    #     "manifest_url": None,  # None for default
    #     "metadata_url": None,  # None for default
    #     "totalsegmentator_organs": ["brain"],
    #     "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    # },
    "tcga-luad": {
        "project": "tcga",
        "cancer_organ": "lung",
        "cancer_type": "lung adenocarcinoma",
        "ONCOTREE_CODE": "LUNG",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-LUAD-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-lusc": {
        "project": "tcga",
        "cancer_organ": "lung",
        "cancer_type": "lung squamous cell carcinoma",
        "ONCOTREE_CODE": "LUNG",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-TCGA-LUSC-01-30-2017.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-ov": {
        "project": "tcga",
        "cancer_organ": "ovary",
        "cancer_type": "ovarian serous cystadenocarcinoma",
        "ONCOTREE_CODE": "OVARY",
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-prad": {
        "project": "tcga",
        "cancer_organ": "prostate",
        "cancer_type": "prostate adenocarcinoma",
        "ONCOTREE_CODE": "PROSTATE",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/doiJNLP-Pz8ET39p.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["prostate"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-read": {
        "project": "tcga",
        "cancer_organ": "rectum",
        "cancer_type": "rectum adenocarcinoma",
        "ONCOTREE_CODE": "BOWEL",
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["colon"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-stad": {
        "project": "tcga",
        "cancer_organ": "stomach",
        "cancer_type": "stomach adenocarcinoma",
        "ONCOTREE_CODE": "STOMACH",
        "manifest_url": None,  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": ["stomach"],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "tcga-ucec": {
        "project": "tcga",
        "cancer_organ": "uterus",
        "cancer_type": "uterine corpus endometrial carcinoma",
        "ONCOTREE_CODE": "UTERUS",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-UCEC-2018-10-24.tcia",  # None for default
        "metadata_url": None,  # None for default
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),  # (min, max) to clip pixel values to before resampling and feature extraction - set to (None, None) for no clipping
    },
    "cptac-ccrcc": {
        "project": "cptac",
        "cancer_organ": "kidney",
        "cancer_type": "clear cell renal cell carcinoma",
        "ONCOTREE_CODE": "KIDNEY",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-CCRCC_v11_20230818.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["kidney_left", "kidney_right"],
        "clip_min,clip_max": (-200, 300),
    },
    # "cptac-ucec": {
    #     "project": "cptac",
    #     "cancer_organ": "uterus",
    #     "cancer_type": "uterine corpus endometrial carcinoma",
    #     "ONCOTREE_CODE": "UTERUS",
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-UCEC_v12_20240405b.tcia",
    #     "metadata_url": None,  #!!! truly doesn't exist - TODO: make a custom one after downloading DICOMs
    #     "totalsegmentator_organs": [],
    #     "clip_min,clip_max": (-200, 300),
    # },
    "cptac-luad": {
        "project": "cptac",
        "cancer_organ": "lung",
        "cancer_type": "lung adenocarcinoma",
        "ONCOTREE_CODE": "LUNG",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-LUAD_v13_20250801.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),
    },
    "cptac-lscc": {
        "project": "cptac",
        "cancer_organ": "lung",
        "cancer_type": "lung squamous cell carcinoma",
        "ONCOTREE_CODE": "LUNG",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-LSCC_v15.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "clip_min,clip_max": (-1000, 400),
    },
    # "cptac-hnscc": {
    #     "project": "cptac",
    #     "cancer_organ": "head_neck",
    #     "cancer_type": "head and neck squamous cell carcinoma",
    #     "ONCOTREE_CODE": "HEAD_NECK",
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-HNSCC_v19_20250226.tcia",
    #     "metadata_url": None,
    #     "totalsegmentator_organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    #     "clip_min,clip_max": (-1000, 400),
    # },
    # "cptac-gbm": {
    #     "project": "cptac",
    #     "cancer_organ": "brain",
    #     "cancer_type": "glioblastoma multiforme",
    #     "ONCOTREE_CODE": "BRAIN",
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-GBM_v16_20240708.tcia",
    #     "metadata_url": None,
    #     "totalsegmentator_organs": ["brain"],
    #     "clip_min,clip_max": (-200, 300),
    # },
    "cptac-aml": {
        "project": "cptac",
        "cancer_organ": "blood",
        "cancer_type": "acute myeloid leukemia",
        "ONCOTREE_CODE": "MYELOID",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/CPTAC-AML_V5_20250415-1.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),
    },
    # "cptac-stad": {
    #     "project": "cptac",
    #     "cancer_organ": "stomach",
    #     "cancer_type": "stomach adenocarcinoma",
    #     "ONCOTREE_CODE": "STOMACH",
    #     "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/CPTAC-STAD_v1_20260304.tcia",
    #     "metadata_url": None,  #!!! truly doesn't exist - TODO: make a custom one after downloading DICOMs
    #     "totalsegmentator_organs": ["stomach"],
    #     "clip_min,clip_max": (-200, 300),
    # },
    "cptac-pda": {
        "project": "cptac",
        "cancer_organ": "pancreas",
        "cancer_type": "pancreatic ductal adenocarcinoma",
        "ONCOTREE_CODE": "PANCREAS",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-PDA_v15_20250226.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": ["pancreas"],
        "clip_min,clip_max": (-200, 300),
    },
    "cptac-cm": {
        "project": "cptac",
        "cancer_organ": "skin",
        "cancer_type": "cutaneous melanoma",
        "ONCOTREE_CODE": "SKIN",
        "manifest_url": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-CM_v11_20240429.tcia",
        "metadata_url": None,
        "totalsegmentator_organs": [],
        "clip_min,clip_max": (-200, 300),
    },
    # add more datasets here as needed
}