"""Download private Hugging Face dataset repos into Kaggle-style mounts.

Each repo lands under ``<inputs>/<name>``, the directory its Kaggle twin
would mount at. `snapshot_download` fetches every file through the Xet
client, which spends about two seconds on a file before its first byte: on a
Colab VM a 415-file sample repo (925 MB) took twelve minutes beside the other
repos, with the GPU billed and idle. The same files over the repo's plain
`resolve` URLs, all repos in one `aria2c` run (`kaggle/aria2.py`, on the
import path the stager sets), took 22 seconds.

Needs `aria2c`, `huggingface_hub` and the Hub token at its default path.
"""

from concurrent.futures import ThreadPoolExecutor

from aria2 import fetch
from huggingface_hub import HfApi, get_token, hf_hub_url

#: Files downloaded at once; the repos are hundreds of shards of a few MB, so
#: most of a file's time is the first byte, and many overlap.
FILES_AT_ONCE = 32


def repo_files(api, repo):
    """Every file of dataset *repo* but the Hub's own `.gitattributes`."""
    info = api.dataset_info(repo)
    return [s.rfilename for s in info.siblings if s.rfilename != ".gitattributes"]


def download(repos, inputs):
    """Download every file of every dataset repo in *repos* under *inputs*; return the mounts."""
    token = get_token()
    if token is None:
        raise FileNotFoundError("no Hugging Face token; upload ~/.cache/huggingface/token first")
    api = HfApi()
    with ThreadPoolExecutor(max_workers=len(repos)) as pool:
        listings = list(pool.map(lambda repo: repo_files(api, repo), repos))
    files = [
        (f"{repo.split('/')[1]}/{name}", hf_hub_url(repo, name, repo_type="dataset"))
        for repo, names in zip(repos, listings, strict=True)
        for name in names
    ]
    fetch(files, inputs, files_at_once=FILES_AT_ONCE, header=f"Authorization: Bearer {token}")
    mounts = []
    for repo, names in zip(repos, listings, strict=True):
        mount = inputs / repo.split("/")[1]
        print(f"{repo}: {len(names)} files", flush=True)
        mounts.append(mount)
    return mounts
