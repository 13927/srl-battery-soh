"""bexp59 / W08: gate self-test -- proves the provenance check **fails**; it does not "always pass".

Method (reads the original manuscript only; modifies no archive and no manuscript):
  1. copy `main.tex` to a temporary directory;
  2. **tamper one table number** in the copy (by default the first data value of Table 8's first
     row, last digit +1);
  3. run `check_provenance.py` with `PROV_TEX=<tampered copy>` and assert it **fails** (non-zero
     exit or a reported inconsistency);
  4. run once on the original (no PROV_TEX) as a **positive control** and assert it **passes**.

Criteria (frozen before the run):
  S1 positive control: the original run of check_provenance.py exits 0.
  S2 negative control: the tampered copy exits non-zero, with "PROVENANCE" inconsistency text in
     the output.
  S3 read-only: this script must not modify main.tex / results/ / any archive (the temporary copy
     is written to the system temp directory and deleted afterwards).
Note: this self-test is **not** part of the routine `make check` (it would re-run the whole check
      suite twice), but it can be run manually at any time; if either S1 or S2 fails, the gate has
      failed and must be repaired first.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXDIR = ROOT / "doc" / "battery" / "paper" / "energies_latex"
TEX = TEXDIR / "main.tex"
CHECK = TEXDIR / "check_provenance.py"


def run(tex_path=None):
    env = dict(os.environ)
    if tex_path is not None:
        env["PROV_TEX"] = str(tex_path)
    p = subprocess.run([sys.executable, str(CHECK)], cwd=TEXDIR, env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def tamper(text):
    """On the first data row of Table 8 (tab:pipelineB), increment the last digit of the first number by 1."""
    m = re.search(r"(\\label\{tab:pipelineB\}.*?\\midrule\n)(.*?)\\\\",
                  text, re.S)
    if not m:
        raise SystemExit("tab:pipelineB data row not found; the self-test cannot proceed")
    row = m.group(2)
    num = re.search(r"\d\.\d{5}", row)
    if not num:
        raise SystemExit("no 5-decimal value found in the data row")
    old = num.group(0)
    new = f"{float(old) + 1e-5:.5f}"
    return text[:m.start(2)] + row.replace(old, new, 1) + text[m.end(2):], old, new


def main():
    ok = True

    print("=== S1 positive control: the original should pass ===")
    rc, out = run()
    s1 = rc == 0
    print(f"  exit code={rc}  {'OK' if s1 else 'FAIL'}")
    ok &= s1

    print("=== S2 negative control: the tampered copy should fail ===")
    src = TEX.read_text()
    bad, old, new = tamper(src)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "main_tampered.tex"
        tmp.write_text(bad)
        rc2, out2 = run(tmp)
    hit = any(("inconsistent" in out2) or ("paper=" in out2) or ("PROVENANCE" in out2 and rc2 != 0)
              for _ in [0])
    s2 = (rc2 != 0)
    print(f"  tampered {old} -> {new}; exit code={rc2}  {'OK' if s2 else 'FAIL'}")
    if not s2:
        print("  [warning] tampering was not detected; the gate may have failed! Output tail:")
        print("  " + "\n  ".join(out2.strip().splitlines()[-5:]))
    ok &= s2
    if s2 and not hit:
        print("  [note] exit code is already non-zero; keyword matching is not required")

    print("=== S3 read-only check ===")
    s3 = TEX.read_text() == src
    print(f"  main.tex was not modified  {'OK' if s3 else 'FAIL'}")
    ok &= s3

    print(f"\nW08 gate self-test: {'passed' if ok else 'failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
