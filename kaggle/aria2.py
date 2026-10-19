"""Download many files over HTTP with one `aria2c` run.

Both kinds of input the kernels stage arrive this way: a Kaggle kernel's
output (`output.py`) and the private Hugging Face dataset repos the Colab legs
mount (`colab/hub.py`). Each file is split across ranged connections, each
segment retried where it broke, and `--continue` picks up a file an earlier
run left partial -- or skips it, when it is already whole.

Needs `aria2c`: Homebrew `aria2` on the Mac, apt `aria2` on the Colab VM.
"""

import subprocess
import tempfile

#: aria2c's ceiling on connections to one host, used for every file.
CONNECTIONS = 16
#: Attempts per segment before a file is given up; each cut connection is one.
ATTEMPTS = 50


def fetch(files, directory, *, files_at_once, header=None):
    """Download every (relative name, url) under *directory*.

    *header* goes with every request. It is written into aria2c's input file
    rather than its command line, so a bearer token never shows in `ps`.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".aria2-input") as listing:
        # aria2c's input-file format: a URI line, then its options indented.
        for name, url in files:
            target = directory / name
            listing.write(f"{url}\n  dir={target.parent}\n  out={target.name}\n")
            if header is not None:
                listing.write(f"  header={header}\n")
        listing.flush()
        subprocess.run(
            [
                "aria2c",
                f"--input-file={listing.name}",
                f"--max-concurrent-downloads={files_at_once}",
                f"--max-connection-per-server={CONNECTIONS}",
                f"--split={CONNECTIONS}",
                "--min-split-size=8M",
                "--continue=true",
                "--auto-file-renaming=false",
                f"--max-tries={ATTEMPTS}",
                "--retry-wait=2",
                "--timeout=60",
                "--file-allocation=none",
                "--summary-interval=60",
                "--show-console-readout=false",
                "--console-log-level=warn",
                "--download-result=hide",
            ],
            check=True,
        )
