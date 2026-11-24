"""Download the Morpho-MNIST data for the 2D Content Alignment experiment (configs/mnist.yaml).

The MNIST experiment needs two Morpho-MNIST perturbation styles -- `thic` (source) and `thin` (target) --
under `data/mnist/`. They are not in the repository (data is git-ignored), so this script fetches them from
the project's GitHub release and verifies every file against a known SHA-256. CIFAR-10 and raw MNIST are
downloaded automatically by torchvision the first time you run the experiment, so they need no action here.

    python scripts/download_data.py                 # into <repo>/data/mnist
    python scripts/download_data.py --data-root /my/data/mnist
    python scripts/download_data.py --force         # re-download even if already present

The data is redistributed from Morpho-MNIST (https://github.com/dccastro/Morpho-MNIST, MIT License,
(c) 2019 Daniel C. Castro et al.); if you use it, cite Castro et al., "Morpho-MNIST: Quantitative Assessment
and Diagnostics for Representation Learning", JMLR 20(178), 2019. The release asset simply mirrors the
official `thic`/`thin` zips; see the bottom of this file for the original download links if the release is
unavailable.
"""
import argparse
import hashlib
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Project GitHub release that hosts the mirrored data asset.
REPO = 'hodakalabizadeh/content-alignment-uda'
TAG = 'morpho-mnist-v1'
ASSET = 'morpho-mnist.zip'
ASSET_SHA256 = '80fcbbcac7f7f0b0ca58e8f82c8439c8702212d82e1de494010650933aa8902b'

# SHA-256 of each file the code reads, relative to the data root (<style>/<split>-...-ubyte.gz).
FILE_SHA256 = {
    'thic/train-images-idx3-ubyte.gz': '08383380ef8c7eeb20e0919378183f96a36a1d7cffb47a9225fd35159893aac1',
    'thic/train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    'thic/t10k-images-idx3-ubyte.gz':  '1321977da1525adeccc821c9519ddd76e53ea82a4a8372698a818d7303c5f1a4',
    'thic/t10k-labels-idx1-ubyte.gz':  'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
    'thin/train-images-idx3-ubyte.gz': '41c899a9d9491a4874b846d21a908cc5f6288336a9ef6d38af051855d457025f',
    'thin/train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    'thin/t10k-images-idx3-ubyte.gz':  '6c2c5988de8316217f89e40e348cf20ce901d61fbd68284288ad2ad30c592e4c',
    'thin/t10k-labels-idx1-ubyte.gz':  'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
}

# Official Morpho-MNIST downloads (Google Drive), for the manual fallback only.
OFFICIAL = {'thic': 'https://drive.google.com/uc?export=download&id=1Uy-SmnEkwq_dptTFuoUtmO9rn2FAbNb8',
            'thin': 'https://drive.google.com/uc?export=download&id=1q3Bfl1oraKZcIPLHnqkU0whnTiz-AVSP'}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def present_and_valid(data_root):
    """True iff every expected file exists under data_root with the right hash."""
    for rel, want in FILE_SHA256.items():
        p = data_root / rel
        if not p.exists() or sha256(p) != want:
            return False
    return True


def download(url, dest):
    print(f'downloading {url}', flush=True)

    def hook(done):
        sys.stdout.write(f'\r  {done / 1e6:6.1f} MB')
        sys.stdout.flush()

    read = 0
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:  # noqa: S310 (fixed https URL)
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            read += len(block)
            hook(read)
    print()


def manual_instructions(data_root):
    lines = [
        '',
        'Could not fetch the release asset. Download the two Morpho-MNIST styles by hand instead:',
        '',
        '  1. Get the official zips from Morpho-MNIST (https://github.com/dccastro/Morpho-MNIST):',
    ]
    lines += [f'       {style}: {url}' for style, url in OFFICIAL.items()]
    lines += [
        '  2. Unpack each so the files land as:',
        f'       {data_root}/thic/{{train,t10k}}-images-idx3-ubyte.gz  (+ matching -labels-idx1-ubyte.gz)',
        f'       {data_root}/thin/{{train,t10k}}-images-idx3-ubyte.gz  (+ matching -labels-idx1-ubyte.gz)',
        '  3. Re-run this script to verify the checksums.',
        '',
    ]
    return '\n'.join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description='Download Morpho-MNIST (thic + thin) for the 2D CA experiment.')
    ap.add_argument('--data-root', type=Path, default=repo_root / 'data' / 'mnist',
                    help='where to put the <style>/ folders (default: <repo>/data/mnist)')
    ap.add_argument('--url', default=f'https://github.com/{REPO}/releases/download/{TAG}/{ASSET}',
                    help='release asset URL to download (override if you host it elsewhere)')
    ap.add_argument('--force', action='store_true', help='re-download even if the files are already present')
    args = ap.parse_args()
    data_root = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)

    if not args.force and present_and_valid(data_root):
        print(f'Morpho-MNIST already present and verified in {data_root}')
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / ASSET
        try:
            download(args.url, zip_path)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
            print(f'error: {e}', file=sys.stderr)
            print(manual_instructions(data_root), file=sys.stderr)
            return 1
        got = sha256(zip_path)
        if got != ASSET_SHA256:
            print(f'error: {ASSET} SHA-256 mismatch\n  expected {ASSET_SHA256}\n  got      {got}', file=sys.stderr)
            print(manual_instructions(data_root), file=sys.stderr)
            return 1
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                # asset lays files out under a top-level 'morpho-mnist/'; strip it so <style>/ sits at data_root
                parts = Path(info.filename).parts
                rel = Path(*parts[1:]) if parts and parts[0] == 'morpho-mnist' else Path(*parts)
                out = data_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(out, 'wb') as dst:
                    dst.write(src.read())

    bad = [rel for rel, want in FILE_SHA256.items() if sha256(data_root / rel) != want]
    if bad:
        print('error: checksum mismatch after extraction for:\n  ' + '\n  '.join(bad), file=sys.stderr)
        return 1
    print(f'Morpho-MNIST (thic + thin) ready in {data_root}')
    print('CIFAR-10 and raw MNIST will be downloaded automatically by torchvision on first run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
