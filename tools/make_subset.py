"""Build an ImageNet-1K class subset (ImageNet-100 / ImageNet-200) as an ImageFolder.

Classes come from a split file (one wnid per line).  By default each class directory
is a symlink into the ImageNet-1K folders; with --short-side, images are instead
re-encoded with the shorter side resized to that many pixels (JPEG quality 92), which
is how the ImageNet-100 copy used here was made (--short-side 256).

    python tools/make_subset.py --imagenet /data/imagenet --split data/splits/imagenet200.txt \
        --out /data/imagenet200
    python tools/make_subset.py --imagenet /data/imagenet --split data/splits/imagenet100.txt \
        --out /data/imagenet100 --short-side 256
"""
import argparse
import os
from multiprocessing import Pool

from PIL import Image

QUALITY = 92


def resize_one(job):
    src, dst, short = job
    if os.path.exists(dst):
        return
    im = Image.open(src).convert('RGB')
    w, h = im.size
    if min(w, h) > short:
        size = (short, round(h * short / w)) if w < h else (round(w * short / h), short)
        im = im.resize(size, Image.BICUBIC)
    im.save(dst, 'JPEG', quality=QUALITY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet', required=True, help='ImageNet-1K root with train/ and val/ class folders')
    ap.add_argument('--split', required=True, help='file with one wnid per line')
    ap.add_argument('--out', required=True)
    ap.add_argument('--short-side', type=int, default=0, help='re-encode at this shorter side instead of linking')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    wnids = [l.strip() for l in open(a.split) if l.strip() and not l.startswith('#')]
    jobs = []
    for split in ('train', 'val'):
        for wnid in wnids:
            src = os.path.join(a.imagenet, split, wnid)
            dst = os.path.join(a.out, split, wnid)
            assert os.path.isdir(src), f'missing {src}'
            if not a.short_side:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
                continue
            os.makedirs(dst, exist_ok=True)
            jobs += [(os.path.join(src, f), os.path.join(dst, f), a.short_side) for f in sorted(os.listdir(src))]
    if jobs:
        with Pool(a.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=256)):
                if (i + 1) % 20000 == 0:
                    print(f'{i + 1}/{len(jobs)}', flush=True)
    print(f'{len(wnids)} classes -> {a.out}')


if __name__ == '__main__':
    main()
