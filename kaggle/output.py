"""Download a Kaggle kernel's output in parallel, resuming whatever broke off.

    python kaggle/output.py <owner>/<kernel> <dir> [--pattern REGEX]

`kaggle kernels output` reads each file in one `requests.get(...).content`, so
a multi-GB checkpoint whose connection drops part-way fails the whole command
with `IncompleteRead` and leaves an empty file behind. The output host also
cuts every connection after about thirty seconds, at well under 1 MB/s each,
so even a resuming single stream needs hours for a 3.5 GB checkpoint. This
lists the same output files through the same API and hands them to `aria2c`
(`aria2.py`, beside this file), a few files at once.

Needs `aria2c` and the `kaggle` package with its credentials: the chain on the
Mac runs it with `uv run --with kaggle`, the Colab staging installs both.
"""

import argparse
import re
from pathlib import Path

from aria2 import fetch
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

#: Files downloaded at once; small files spend most of their time waiting for
#: the host's first byte, so several overlap.
FILES_AT_ONCE = 4


def output_files(api, kernel):
    """Every (relative name, signed url) the kernel's latest session wrote."""
    owner, slug = kernel.split("/")
    files = []
    token = ""
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 200
            request.page_token = token
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
            files += [(item.file_name, item.url) for item in response.files or []]
            token = response.next_page_token
            if not token:
                return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kernel", help="<owner>/<kernel>")
    parser.add_argument("dir", type=Path)
    parser.add_argument("--pattern", help="regex a file's relative name must match")
    args = parser.parse_args()
    pattern = re.compile(args.pattern) if args.pattern else None

    api = KaggleApi()
    api.authenticate()
    files = output_files(api, args.kernel)
    wanted = [(name, url) for name, url in files if pattern is None or pattern.search(name)]
    if not wanted:
        raise SystemExit(f"{args.kernel} has no output matching {args.pattern!r}")
    fetch(wanted, args.dir.resolve(), files_at_once=FILES_AT_ONCE)
    for name, _ in wanted:
        print(f"{name}: {(args.dir / name).stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
