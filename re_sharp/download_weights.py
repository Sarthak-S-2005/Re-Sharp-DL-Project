"""Optional: download the pretrained weights BEFORE the demo (so nothing is downloaded live).

Usage:
    python download_weights.py          # both 2x and 4x
    python download_weights.py 2        # only 2x
"""
import sys

from models.model_manager import MODEL_SPECS, ModelUnavailableError, ensure_weights


def main() -> int:
    scales = [int(a) for a in sys.argv[1:]] or [2, 4]
    for s in scales:
        if s not in MODEL_SPECS:
            print(f"Unknown scale {s}. Choose from {list(MODEL_SPECS)}")
            return 1
        spec = MODEL_SPECS[s]
        print(f"Downloading {spec.name} ...")
        last = {"p": -1}

        def cb(p):
            pct = int(p * 100)
            if pct // 10 != last["p"] // 10:
                print(f"  {pct}%")
            last["p"] = pct

        try:
            path = ensure_weights(s, cb)
            print(f"  OK -> {path}")
        except ModelUnavailableError as exc:
            print(f"  FAILED:\n{exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
