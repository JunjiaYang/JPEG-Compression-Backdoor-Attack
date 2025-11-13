"""Download and extract the GTSRB dataset."""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"


def download(destination: str = "data/raw/gtsrb") -> Path:
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "gtsrb.zip"
    if not archive.exists():
        urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest


if __name__ == "__main__":  # pragma: no cover
    download()
