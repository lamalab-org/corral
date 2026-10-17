"""Where the trial-wide ledger lives for a trial and for its branches."""

from inference_opt.tools import _shared_ledger

from corral.runtime.permissions import NODE_WORKSPACE_DIR


def test_a_trial_and_its_branches_find_the_same_ledger(tmp_path):
    trial = tmp_path / "trial-1"
    branch = trial / NODE_WORKSPACE_DIR / "node-a"
    branch.mkdir(parents=True)

    main = _shared_ledger(str(trial))
    node = _shared_ledger(str(branch))

    assert main.path == node.path
    assert main.branch == ""
    assert node.branch == "node-a"
    # Beside the trial's workspace, not inside it, out of the agent's reach.
    assert not main.path.is_relative_to(trial)


def test_without_a_workspace_there_is_no_shared_ledger():
    assert _shared_ledger("") is None
