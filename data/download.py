"""Download RadioGalaxyNET (CSIRO DAP collection 61068, DOI 10.25919/btk3-vx79) into data/.

  python data/download.py            -> data/RadioGalaxyNET/{annotations,train,val,test}

Lists the collection through the DAP REST API, downloads every file in parallel, verifies byte
sizes and re-downloads mismatches. Safe to re-run. The dataset is CC BY-NC-SA 4.0 (see the DAP
page); this repo does not redistribute it.
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LISTING = "https://data.csiro.au/dap/ws/v2/collections/61068/data"
OUT = Path(__file__).resolve().parent


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(f):
    dst = OUT / f["filename"]  # filenames already start with RadioGalaxyNET/
    if dst.exists() and dst.stat().st_size == f["fileSize"]:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for _ in range(4):
        try:
            data = get(f["link"]["href"], timeout=60)
            if len(data) == f["fileSize"]:
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
                return None
            err = f"size {len(data)} != {f['fileSize']}"
        except Exception as e:  # noqa: BLE001
            err = e
    return f"{f['filename']}: failed ({err})"


def main():
    files = [f for f in json.loads(get(LISTING))["file"]
             if not f["filename"].endswith(".DS_Store")]
    with ThreadPoolExecutor(16) as ex:
        errs = [e for e in ex.map(fetch, files) if e]
    print(f"{len(files)} files, {len(errs)} errors")
    for e in errs:
        print(e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
