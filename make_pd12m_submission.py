from pathlib import Path
from io import BytesIO
import zipfile
import shutil
import hashlib
import re
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# =========================
# Paths
# =========================

WORK_DIR = Path("Path/to/your/workdir")
NEWS_CSV = WORK_DIR / "newsimages_test_and_evaluation_26_v1.0/news_articles_combined.csv"
PD12M_DIR = Path("Path/to/your/PD12M/metadata")

GROUP_NAME = "AIINS_01"
APPROACH_NAME = "PD12MTFIDF"

OUTPUT_DIR = WORK_DIR / f"{GROUP_NAME}_{APPROACH_NAME}"
ZIP_PATH = WORK_DIR / f"{GROUP_NAME}_Submission.zip"
LOG_PATH = WORK_DIR / "pd12m_submission_log.csv"

TARGET_W = 460
TARGET_H = 260


# =========================
# Speed / quality settings
# =========================

NUM_PARQUET_FILES = 125
SAMPLE_PER_PARQUET = 100000

TOP_K = 150
RANDOM_SEED = 5262


# =========================
# Helper functions
# =========================

def clean_article_id(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()

    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    return s


def read_news_csv(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read news CSV: {last_error}")


def select_parquet_files(pd12m_dir, num_files):
    files = sorted(pd12m_dir.glob("*.parquet"))

    if not files:
        raise RuntimeError(f"No parquet files found in {pd12m_dir}")

    if len(files) <= num_files:
        return files

    # Pick files evenly across 0~124, not just the first few files.
    indices = np.linspace(0, len(files) - 1, num_files, dtype=int)
    selected = [files[i] for i in indices]

    return selected


def load_pd12m_candidates():
    selected_files = select_parquet_files(PD12M_DIR, NUM_PARQUET_FILES)

    print("=== Selected PD12M parquet files ===")
    for p in selected_files:
        print(p.name)

    all_parts = []

    for file_idx, parquet_path in enumerate(tqdm(selected_files, desc="Loading PD12M metadata")):
        df = pd.read_parquet(
            parquet_path,
            columns=["id", "url", "caption", "width", "height", "mime_type"]
        )

        df["id"] = df["id"].fillna("").astype(str)
        df["url"] = df["url"].fillna("").astype(str)
        df["caption"] = df["caption"].fillna("").astype(str)
        df["mime_type"] = df["mime_type"].fillna("").astype(str)

        df = df[df["caption"].str.len() > 10]
        df = df[df["url"].str.startswith("http")]
        df = df[df["width"] >= TARGET_W]
        df = df[df["height"] >= TARGET_H]
        df = df[df["mime_type"].str.contains("image", case=False, na=False)]

        if len(df) > SAMPLE_PER_PARQUET:
            df = df.sample(
                n=SAMPLE_PER_PARQUET,
                random_state=RANDOM_SEED + file_idx
            )

        all_parts.append(df)

    pd12m = pd.concat(all_parts, ignore_index=True)
    pd12m = pd12m.drop_duplicates(subset=["url"]).reset_index(drop=True)

    print()
    print(f"[INFO] PD12M candidate count: {len(pd12m)}")

    if len(pd12m) == 0:
        raise RuntimeError("No usable PD12M candidates loaded.")

    return pd12m


def center_crop_resize(img):
    img = img.convert("RGB")
    w, h = img.size

    scale = max(TARGET_W / w, TARGET_H / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - TARGET_W) // 2
    top = (new_h - TARGET_H) // 2
    right = left + TARGET_W
    bottom = top + TARGET_H

    return img.crop((left, top, right, bottom))


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=25)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content))
    img.load()

    return img


def make_fallback_card(output_path, article_id, title):
    seed_text = f"{article_id}_{title}"
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()

    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)

    bg = (max(35, r), max(35, g), max(35, b))

    img = Image.new("RGB", (TARGET_W, TARGET_H), bg)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, TARGET_W, 54], fill=(20, 20, 20))
    draw.text((18, 18), "NEWS IMAGE", fill=(255, 255, 255))

    title = str(title)
    title = title[:90]

    draw.text((18, 92), title, fill=(255, 255, 255))
    draw.text((18, 222), f"article_id: {article_id}", fill=(255, 255, 255))

    img.save(output_path, "PNG")


def make_output_path(article_id):
    filename = f"{article_id}_{GROUP_NAME}_{APPROACH_NAME}.png"
    return OUTPUT_DIR / filename


def validate_outputs(expected_article_ids):
    png_files = sorted(OUTPUT_DIR.glob("*.png"))

    expected_count = len(expected_article_ids)
    actual_count = len(png_files)

    print()
    print("=== Validation ===")
    print(f"Expected PNG count: {expected_count}")
    print(f"Actual PNG count:   {actual_count}")

    if actual_count != expected_count:
        raise RuntimeError("PNG count mismatch.")

    existing_names = set(p.name for p in png_files)

    missing = []
    for article_id in expected_article_ids:
        expected_name = f"{article_id}_{GROUP_NAME}_{APPROACH_NAME}.png"
        if expected_name not in existing_names:
            missing.append(expected_name)

    if missing:
        print("Missing examples:")
        for x in missing[:20]:
            print(x)
        raise RuntimeError("Some article_id outputs are missing.")

    bad_size = []

    for p in png_files:
        try:
            with Image.open(p) as img:
                if img.size != (TARGET_W, TARGET_H):
                    bad_size.append(str(p))
        except Exception:
            bad_size.append(str(p))

    print(f"Bad size count: {len(bad_size)}")

    if bad_size:
        print("Bad size examples:")
        for x in bad_size[:20]:
            print(x)
        raise RuntimeError("Some PNG files are not 460x260.")

    print("Validation passed.")


def create_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for png_path in sorted(OUTPUT_DIR.glob("*.png")):
            arcname = png_path.relative_to(WORK_DIR)
            zf.write(png_path, arcname)

    print()
    print(f"[INFO] ZIP created: {ZIP_PATH}")


# =========================
# Main workflow
# =========================

def main():
    start_time = time.time()

    print("=== NewsImages PD12M TF-IDF submission ===")
    print(f"Work dir: {WORK_DIR}")
    print(f"News CSV: {NEWS_CSV}")
    print(f"PD12M dir: {PD12M_DIR}")
    print(f"Group name: {GROUP_NAME}")
    print(f"Approach name: {APPROACH_NAME}")

    print()
    print("=== Load NewsImages test articles ===")

    news_df = read_news_csv(NEWS_CSV)

    test_df = news_df[
        news_df["article_url"].isna() &
        news_df["image_id"].isna()
    ].copy()

    test_df["article_id"] = test_df["article_id"].apply(clean_article_id)
    test_df["article_title"] = test_df["article_title"].fillna("").astype(str)

    test_df = test_df[test_df["article_id"] != ""].reset_index(drop=True)

    print(f"[INFO] Test article count: {len(test_df)}")

    if len(test_df) != 800:
        print("[WARN] Test article count is not 800. Continue anyway.")

    print()
    print("First 5 test articles:")
    print(test_df[["article_id", "article_title"]].head(5))

    print()
    print("=== Load PD12M candidates ===")

    pd12m = load_pd12m_candidates()

    print()
    print("=== Prepare output folder ===")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output folder: {OUTPUT_DIR}")

    print()
    print("=== Build TF-IDF features ===")

    candidate_texts = pd12m["caption"].astype(str).tolist()
    test_texts = test_df["article_title"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000
    )

    all_texts = candidate_texts + test_texts
    features = vectorizer.fit_transform(all_texts)

    candidate_features = features[:len(candidate_texts)]
    test_features = features[len(candidate_texts):]

    print(f"[INFO] Candidate feature shape: {candidate_features.shape}")
    print(f"[INFO] Test feature shape: {test_features.shape}")

    print()
    print("=== Retrieve, download, and save images ===")

    used_urls = set()
    logs = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test articles"):
        article_id = row["article_id"]
        title = row["article_title"]

        output_path = make_output_path(article_id)

        scores = (candidate_features @ test_features[i].T).toarray().ravel()

        if len(scores) <= TOP_K:
            top_indices = np.argsort(-scores)
        else:
            rough = np.argpartition(-scores, TOP_K)[:TOP_K]
            top_indices = rough[np.argsort(-scores[rough])]

        success = False
        last_error = ""

        for idx in top_indices:
            idx = int(idx)
            cand = pd12m.iloc[idx]

            url = str(cand["url"])
            pd12m_id = str(cand["id"])
            caption = str(cand["caption"])
            score = float(scores[idx])

            if url in used_urls:
                continue

            try:
                img = download_image(url)
                img = center_crop_resize(img)
                img.save(output_path, "PNG")

                used_urls.add(url)

                logs.append({
                    "article_id": article_id,
                    "article_title": title,
                    "method": "pd12m_tfidf",
                    "pd12m_id": pd12m_id,
                    "pd12m_url": url,
                    "pd12m_caption": caption,
                    "score": score,
                    "output_file": output_path.name
                })

                success = True
                break

            except Exception as e:
                last_error = str(e)
                continue

        if not success:
            make_fallback_card(output_path, article_id, title)

            logs.append({
                "article_id": article_id,
                "article_title": title,
                "method": "fallback_card",
                "pd12m_id": "",
                "pd12m_url": "",
                "pd12m_caption": "",
                "score": 0.0,
                "output_file": output_path.name,
                "last_error": last_error
            })

    print()
    print("=== Save log ===")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] Log saved: {LOG_PATH}")

    method_counts = log_df["method"].value_counts()
    print()
    print("Method counts:")
    print(method_counts)

    expected_article_ids = test_df["article_id"].tolist()

    validate_outputs(expected_article_ids)
    create_zip()

    elapsed = time.time() - start_time

    print()
    print("=== Done ===")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Submission ZIP: {ZIP_PATH}")
    print(f"Log CSV: {LOG_PATH}")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
