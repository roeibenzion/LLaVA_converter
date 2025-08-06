#!/usr/bin/env python3
"""
download_llava_v1_5_data.py
Fast downloader for LLaVA‑v1.5 visual‑instruction‑tuning mixture
(images + llava_v1_5_mix665k.json).

usage:  python download_llava_v1_5_data.py --workers 16 --root ./playground/data
requirements:
    pip install aiohttp aiofiles tqdm
(optional) install aria2c for maximal speed
"""

import argparse, asyncio, json, os, pathlib, shutil, subprocess, sys, zipfile
from functools import partial
from typing import Dict, List
from tqdm.asyncio import tqdm_asyncio

# ---------------------------------------------------------------------------
# 1.  URLs & target layout  (same hierarchy the training scripts expect)
#     ref: GitHub README “Visual Instruction Tuning” and setup_finetune.sh
# ---------------------------------------------------------------------------
DATA_SPEC: List[Dict[str, str]] = [
    # annotation (1.03 GB, Hugging Face LFS)
    {
        "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json",
        "path": "llava_v1_5_mix665k.json",
        "is_zip": False,
    },
    # COCO
    {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "path": "images/coco/train2017/train2017.zip",
        "is_zip": True,
    },
    # GQA
    {
        "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        "path": "images/gqa/images/images.zip",
        "is_zip": True,
    },
    # TextVQA
    {
        "url": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        "path": "images/textvqa/train_images/train_val_images.zip",
        "is_zip": True,
    },
    # Visual Genome – images part 1 & part 2
    {
        "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "path": "images/vg/VG_100K/images.zip",
        "is_zip": True,
    },
    {
        "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
        "path": "images/vg/VG_100K_2/images2.zip",
        "is_zip": True,
    },
    # OCR‑VQA      (mirrored on HF for convenience; avoids Google‑Drive quota)
    {
        "url": "https://huggingface.co/datasets/weizhiwang/llava_v15_instruction_images/resolve/main/ocr_vqa_images_llava_v15.zip",
        "path": "images/ocr_vqa/images/ocr_vqa_images.zip",
        "is_zip": True,
    },
]

# ---------------------------------------------------------------------------
# 2.  Helpers
# ---------------------------------------------------------------------------
def aria2c_available() -> bool:
    return shutil.which("aria2c") is not None


def ensure_parent(file_path: pathlib.Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)


def aria2c_download(url: str, out_path: pathlib.Path, connections: int = 16):
    ensure_parent(out_path)
    cmd = [
        "aria2c",
        "-x", str(connections),     # max connections per server
        "-s", str(connections),     # split into this many chunks
        "-k", "1M",                 # chunk size
        "-o", out_path.name,
        "-d", str(out_path.parent),
        url,
    ]
    subprocess.run(cmd, check=True)


# -------- async fallback ----------------------------------------------------
async def aio_download(session, url: str, out_path: pathlib.Path, chunk_size=1 << 20):
    ensure_parent(out_path)
    async with session.get(url) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        pbar = tqdm_asyncio(total=total, unit="B", unit_scale=True, desc=out_path.name)
        async with aiofiles.open(out_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(chunk_size):
                await f.write(chunk)
                await pbar.update(len(chunk))
        await pbar.close()


async def aio_download_all(jobs, workers):
    import aiohttp, aiofiles  # imported lazily
    connector = aiohttp.TCPConnector(limit=workers)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await asyncio.gather(*(aio_download(session, *job) for job in jobs))


# -------- extraction --------------------------------------------------------
def extract_zip(zip_path: pathlib.Path):
    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    zip_path.unlink()  # delete .zip after success


# ---------------------------------------------------------------------------
# 3.  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./playground/data", help="destination root folder")
    parser.add_argument("--workers", type=int, default=16, help="concurrent connections")
    args = parser.parse_args()
    root = pathlib.Path(args.root).expanduser().resolve()

    jobs = []
    for item in DATA_SPEC:
        target = root / item["path"]
        if target.exists():
            print(f"✓ {target} already present, skipping.")
            continue
        jobs.append((item["url"], target, item["is_zip"]))

    if not jobs:
        print("All files already present – nothing to do.")
        return

    # -- download ------------------------------------------------------------
    if aria2c_available():
        print(">> aria2c detected – using fastest multi‑connection mode.")
        for url, path, _ in jobs:
            aria2c_download(url, path, connections=args.workers)
    else:
        print(">> aria2c not found – falling back to aiohttp downloader.")
        asyncio.run(aio_download_all([(url, path) for url, path, _ in jobs],
                                     workers=args.workers))

    # -- extract -------------------------------------------------------------
    for _, path, is_zip in jobs:
        if is_zip:
            extract_zip(path)

    print("\n✅  LLaVA‑v1.5 visual‑instruction data ready under", root)


if __name__ == "__main__":
    main()
