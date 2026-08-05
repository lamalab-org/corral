import sys
import traceback
from pathlib import Path

import cloudpickle

from corral.backend.executors import render_result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:  # pragma: no cover - defensive; the executor always passes 2
        sys.stderr.write("usage: _job_worker <input.pkl> <output.pkl>\n")
        return 2
    in_path, out_path = Path(argv[0]), Path(argv[1])

    with in_path.open("rb") as fh:
        tool, call_arguments = cloudpickle.load(fh)

    try:
        payload = {"ok": True, "result": render_result(tool.execute(**call_arguments))}
    except Exception as exc:  # captured for the parent, never re-raised
        payload = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    with out_path.open("wb") as fh:
        cloudpickle.dump(payload, fh)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
