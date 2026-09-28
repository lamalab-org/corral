"""The MD log reader needs its Polars Python runtime at execution time."""

from corral_md.tools import get_nth_run_log


def test_get_nth_run_log_parses_thermo_data(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text(
        "LAMMPS (19 Nov 2024)\n"
        "run 10\n"
        "Per MPI rank memory allocation (min/avg/max) = 2.0 | 2.0 | 2.0 Mbytes\n"
        "   Step          Temp          E_pair         E_mol          TotEng         Press\n"
        "         0   300           -10             0             -10              1\n"
        "        10   310            -9             0              -9              1.2\n"
        "Loop time of 0.1 on 1 procs for 10 steps with 1 atoms\n"
    )

    result = get_nth_run_log.execute(path=str(log))

    assert "Step" in result
    assert "310" in result
    assert "Failed to parse" not in result
