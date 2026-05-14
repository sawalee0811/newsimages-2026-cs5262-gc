# NewsImages Challenge 2026 - AIINS 01
This repository contains our simple submission workflow for the NewsImages Challenge 2026.

## Method

We use a lightweight image retrieval approach based on PD12M metadata.

For each NewsImages test article, we use the article title as a text query. We compare the title with PD12M image captions using TF-IDF. The image with the highest similarity score is downloaded, resized, center-cropped to 460x260 pixels, and renamed into the required submission format.

## Dataset

- NewsImages test/evaluation dataset: provides test article IDs and article titles.
- PD12M metadata: provides image captions and image URLs.

The datasets are not included in this repository.

PD12M metadata
Install the huggingface-cli utility (via pip). You may then use the following command:

```bash
huggingface-cli download Spawning/PD12M --repo-type dataset --local-dir metadata --include "metadata/*"
```

## Requirements

```bash
conda create -n newsimg python=3.10 -y
conda activate newsimg
conda install -c conda-forge pandas pyarrow scikit-learn pillow tqdm requests -y
```

## Run
```bash
python make_pd12m_submission.py
```

## Output
The script generates:

```bash
AIINS_01_Submission.zip
└── AIINS_01_PD12MTFIDF/
    ├── [article_id]_AIINS_01_PD12MTFIDF.png
    └── ...
```

## Result
please see [here](https://drive.google.com/file/d/1ctqYzjAkifGx71Zw-p7V79HmT5GwLIC5/view?usp=sharing)

