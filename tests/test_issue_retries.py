"""Bookkeeping and incident detection must not silently add benchmark attempts."""
import gzip
import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import retry_issues_campaign as campaign
import retry_issues_worker as worker
import retry_issues_health as health


@pytest.mark.parametrize('agent', ['react', 'openhands'])
def test_kimi_retry_uses_only_together_and_no_reasoning_override(agent):
    item = {'agent': agent, 'model': 'kimi-k3', 'slot': 3, 'session_id': 'original-session',
            'task': 'task', 'level': 1, 'environment': 'stargazer',
            'agent_runtime': {'name': agent, 'model': 'openrouter/moonshotai/kimi-k3',
                              'reasoning_effort': 'medium', 'options': {'reasoning_effort': 'medium'}},
            'environment_runtime': {'name': 'stargazer', 'options': {}}}
    request = campaign.request_for(item, 2, 'image')
    assert request['benchmark_run_id'] == 'original-session'
    assert request['trial_index'] == 3
    runtime = request['agent_runtime']
    assert runtime['reasoning_effort'] is None
    assert 'reasoning_effort' not in runtime['options']
    key = 'litellm_extra_body' if agent == 'openhands' else 'extra_body'
    assert runtime['options'][key]['provider'] == campaign.PROVIDER
    if agent == 'openhands':
        assert runtime['options']['num_retries'] == 12
        assert runtime['options']['retry_max_wait'] == 300
    else:
        assert runtime['options']['llm_retry_policy'] == {
            'max_attempts': 12, 'min_wait_s': 15, 'max_wait_s': 300, 'multiplier': 15}


def test_kimi_concurrency_limit_leaves_capacity_for_other_models():
    inventory = [{'id': 'one', 'model': 'kimi-k3'}, {'id': 'two', 'model': 'kimi-k3'},
                 {'id': 'three', 'model': 'gpt-5.6-sol'}]
    states = {'one': {'state': 'running'}, 'two': {'state': 'running'},
              'three': {'state': 'queued'}}
    assert not campaign.provider_room(inventory[0], inventory, states)
    assert campaign.provider_room(inventory[2], inventory, states)
    states['two']['state'] = 'replaced'
    assert campaign.provider_room(inventory[0], inventory, states)


def test_replacement_keeps_slot_count_session_and_recomputes_metrics(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    run = tmp_path / 'campaign'
    run.mkdir()
    source = root / 'reports/sample/level_1/react__kimi-k3/report.json.gz'
    source.parent.mkdir(parents=True)
    trials = [{'trial_id': f'original:task:{i}', 'score': 0., 'success': False,
               'duration': 1., 'token_usage': {}, 'tool_calls': [], 'error_message': None} for i in range(5)]
    original = trials[3]
    report = {'metadata': {'benchmark': {'k_values': [1,2,3,4,5]}}, 'task_results': {'task': {'trials': trials}}}
    source.write_bytes(gzip.compress(json.dumps(report).encode()))
    attempt = run / 'artifacts/attempt/state'
    attempt.mkdir(parents=True)
    replacement = {**original, 'trial_id': 'original:task:retry-3-1', 'score': 1., 'success': True}
    (attempt / 'report.json').write_text(json.dumps({'task_results': {'task': {'trials': [replacement]}}}))
    (attempt / 'request.json').write_text('{}')
    monkeypatch.setattr(campaign, 'ROOT', root)
    monkeypatch.setattr(campaign, 'CAMPAIGN', run)
    calls = []
    def langfuse(method, trace):
        calls.append((method, trace))
        return {'sessionId': 'original'}
    monkeypatch.setattr(campaign, 'langfuse_request', langfuse)
    item = {'id': 'test', 'report': str(source.relative_to(root)), 'task': 'task', 'slot': 3,
            'session_id': 'original', 'original_execution_id': original['trial_id'], 'original_trace_id': 'old-trace',
            'original_trial_sha256': hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
            'reasons': ['disk_exhaustion']}
    status = {'root': str(attempt.parent)}
    campaign.merge(item, status)
    final = campaign.load_report(source)
    assert len(final['task_results']['task']['trials']) == 5
    assert final['task_results']['task']['trials'][3] == replacement
    assert final['metrics']['Average Score'] == pytest.approx(.2)
    assert final['metrics']['Pass@1'] == pytest.approx(.2)
    assert source.with_name('report.json').is_symlink()
    assert json.loads(source.with_name('report.json').read_text()) == final
    assert calls[-1] == ('DELETE', 'old-trace')
    assert status['state'] == 'replaced'


@pytest.mark.parametrize('kind,error,stops', [
    ('tool.failed', "cannot pickle 're.Match' object", True),
    ('tool.completed', 'RecursionError: maximum recursion depth exceeded', False),
])
def test_watchdog_distinguishes_checkpoint_failure_from_agent_python_error(tmp_path, monkeypatch, kind, error, stops):
    with sqlite3.connect(tmp_path / 'commits.sqlite3') as db:
        db.execute('CREATE TABLE commits (sequence INTEGER,event_type TEXT,event_json TEXT)')
        db.execute('INSERT INTO commits VALUES (0,?,?)', ('tool.started', json.dumps({'invocation_id': 'test', 'tool_name': 'PythonREPL'})))
        db.execute('INSERT INTO commits VALUES (1,?,?)', (kind, json.dumps({'invocation_id': 'test', 'error': error})))
    class Stop:
        counter = 0
        def wait(self, _):
            self.counter += 1
            return self.counter > 1
    signals = []
    monkeypatch.setattr(worker.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    worker.watch_infrastructure(tmp_path, Stop())
    assert bool(signals) == stops
    assert (tmp_path / 'watchdog-failure.json').exists() == stops
    assert bool(health.inspect(tmp_path)['failures']) == stops


def test_report_detection_does_not_retry_agent_recursion_or_excluded_outcomes():
    trial = {'error_message': 'Iteration limit reached', 'tool_calls': [
        {'tool_name': 'PythonREPL', 'status': 'completed',
         'result': 'RecursionError: maximum recursion depth exceeded'},
        {'tool_name': 'predict_ir', 'status': 'failed', 'result': 'IR prediction failed'},
    ]}
    assert campaign.trial_failures(trial) == []
    trial['tool_calls'][0]['status'] = 'failed'
    assert campaign.trial_failures(trial) == ['maximum recursion depth exceeded']


def test_provider_failure_is_held_for_review():
    assert campaign.trial_failures({'error_message': 'AuthenticationError: token rejected'})


@pytest.mark.parametrize('message,incident', [
    ('Agent reached maximum iterations limit (50).', False),
    ("Claude Agent SDK run did not succeed: subtype='error_max_turns', stop_reason='tool_use', result=None", False),
    ('Incorrect answer format', False),
    ("Client error '429 Too Many Requests'. For more information check: https://example.org", True),
    ('Invalid prompt: your prompt was flagged by the provider.', True),
])
def test_failure_classifier_preserves_iteration_outcomes_and_provider_errors(message, incident):
    assert bool(campaign.trial_failures({'error_message': message})) == incident


def test_langfuse_session_lookup_omits_large_observation_payloads(monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, 'dotenv_values', lambda _: {
        'LANGFUSE_BASE_URL': 'https://example.test', 'LANGFUSE_PUBLIC_KEY': 'public',
        'LANGFUSE_SECRET_KEY': 'test-secret'})
    def request(method, url, **kwargs):
        calls.append((method, kwargs.get('params')))
        if method == 'GET':
            assert kwargs['params'] == {'fields': 'core'}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'sessionId': 'original-session'})
    monkeypatch.setattr(campaign.requests, 'request', request)
    assert campaign.langfuse_request('GET', 'large-trace')['sessionId'] == 'original-session'
    campaign.langfuse_request('DELETE', 'old-trace')
    assert calls == [('GET', {'fields': 'core'}), ('DELETE', None)]


@pytest.mark.parametrize('tool,output,incident', [
    ('PythonREPL', "Error Traceback:\nModuleNotFoundError: No module named 'astropy'", False),
    ('map_reaction_smiles', "ModuleNotFoundError: No module named 'rxnmapper'", True),
    ('PythonREPL', 'Error Traceback:\nOSError: Disk quota exceeded', True),
    ('PythonREPL', 'Error Traceback:\nRecursionError: maximum recursion depth exceeded', False),
    ('execute_python_script', 'Local GPU Python requires an available CUDA GPU', True),
    ('execute_python_script', "Python script failed: Couldn't find MACE model files: /workspace/models/student.model", False),
])
def test_similar_infrastructure_failures_are_distinguished_from_agent_errors(tool, output, incident):
    assert bool(health.tool_incidents(tool, True, output)) == incident


@pytest.mark.parametrize('own_count,allowed', [(0, True), (9, True), (10, False)])
def test_parallel_limit_counts_retry_campaign_and_allows_other_jobs(monkeypatch, own_count, allowed):
    entries = []
    for n in range(27 + own_count):
        labels = {'corral.task_id': str(n)}
        if n >= 27:
            labels['corral.retry_campaign'] = '20261016'
        entries.append({'Name': '/task-' + str(n), 'Config': {'Labels': labels},
                        'HostConfig': {'NanoCpus': campaign.TRIAL_CPUS * 1e9, 'Memory': campaign.TRIAL_MEMORY}})
    monkeypatch.setattr(campaign, 'docker', lambda *args, **kwargs: SimpleNamespace(
        stdout='container-ids' if args[0] == 'ps' else json.dumps(entries)))
    monkeypatch.setattr(campaign, 'container_usage', lambda _: (0, 0))
    monkeypatch.setattr(campaign.psutil, 'cpu_percent', lambda **_: 5)
    monkeypatch.setattr(campaign.psutil, 'cpu_count', lambda: 64)
    monkeypatch.setattr(campaign.psutil, 'virtual_memory', lambda: SimpleNamespace(total=256 * 2**30, available=240 * 2**30))
    monkeypatch.setattr(campaign.psutil, 'disk_usage', lambda path: SimpleNamespace(total=1000, free=193 if path == '/' else 400))
    capacity = campaign.resources()
    assert capacity['active_trials'] == 27 + own_count
    assert capacity['active_retry_trials'] == own_count
    assert capacity['can_launch'] == allowed


def test_parallel_increase_still_reserves_twenty_percent_headroom(monkeypatch):
    monkeypatch.setattr(campaign, 'docker', lambda *args, **kwargs: SimpleNamespace(stdout='', returncode=0))
    monkeypatch.setattr(campaign.psutil, 'cpu_count', lambda: 64)
    monkeypatch.setattr(campaign.psutil, 'cpu_percent', lambda **_: 78)
    monkeypatch.setattr(campaign.psutil, 'virtual_memory', lambda: SimpleNamespace(total=256 * 2**30, available=240 * 2**30))
    monkeypatch.setattr(campaign.psutil, 'disk_usage', lambda path: SimpleNamespace(total=1000, free=400))
    capacity = campaign.resources()
    assert not capacity['can_launch']
    assert any('CPU' in reason for reason in capacity['waiting_reasons'])
    monkeypatch.setattr(campaign.psutil, 'cpu_percent', lambda **_: 0)
    monkeypatch.setattr(campaign.psutil, 'virtual_memory', lambda: SimpleNamespace(total=256 * 2**30, available=65 * 2**30))
    capacity = campaign.resources()
    assert not capacity['can_launch']
    assert any('memory' in reason for reason in capacity['waiting_reasons'])


@pytest.mark.parametrize('agent', ['ai-scientist', 'claude-code', 'react'])
def test_claude_retry_preserves_original_session_slot_and_budgets(agent):
    item = {'agent': agent, 'model': 'claude-opus-5-5', 'slot': 2, 'session_id': f'level2-{agent}-original',
            'task': 'task', 'level': 2, 'environment': 'retrosynthesis',
            'agent_runtime': {'name': agent, 'model': 'anthropic/claude-opus-5-5', 'options': {}},
            'environment_runtime': {'name': 'retrosynthesis', 'options': {}}}
    request = campaign.request_for(item, 2, 'image')
    assert request['benchmark_run_id'] == item['session_id']
    assert request['trial_index'] == item['slot']
    assert request['agent_runtime']['options']['reasoning_effort'] == 'medium'
    assert request['max_iterations'] == (100 if agent == 'ai-scientist' else 50)
    if agent == 'ai-scientist':
        assert request['agent_runtime']['options']['config']['max_tool_calls'] == 256
    assert request['sandbox']['docker']['cpus'] == 4
    assert request['sandbox']['docker']['memory'] == '18g'


def test_resource_inspection_failure_holds_launches_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(campaign, 'docker', lambda *args, **kwargs: SimpleNamespace(
        stdout='gone-container' if args[0] == 'ps' else '', returncode=0 if args[0] == 'ps' else 1))
    assert campaign.resources()['can_launch'] is False


def test_additional_provider_retries_are_bounded_and_do_not_repeat_valid_outcomes(monkeypatch):
    calls = []
    def requeue(item, status, note):
        calls.append(status['attempt'])
        status.update(state='queued', repairs=[note])
    monkeypatch.setattr(campaign, 'requeue', requeue)
    policy = {'last_attempt': 4, 'cooldown_seconds': 30,
              'failure_markers': ['Invalid prompt: your prompt was flagged']}
    failure = ['Unexpected runtime failure: Invalid prompt: your prompt was flagged by provider']
    item = {'id': 'requested-trial'}
    status = {'state': 'unresolved', 'attempt': 1, 'failures': failure}
    for attempt in (1, 2, 3):
        status.update(state='needs_repair', attempt=attempt)
        assert campaign.retry_authorized_provider_failure(item, status, policy)
        assert status['state'] == 'queued'
        assert status['retry_not_before'] > 0
    status.update(state='needs_repair', attempt=4)
    assert not campaign.retry_authorized_provider_failure(item, status, policy)
    assert status['state'] == 'unresolved'
    assert calls == [1, 2, 3]
    status.update(state='replaced', attempt=2, score=0.0)
    assert not campaign.retry_authorized_provider_failure(item, status, policy)
    status.update(state='needs_repair', attempt=2, failures=['cannot pickle checkpoint'])
    assert not campaign.retry_authorized_provider_failure(item, status, policy)
    status.update(failures=failure)
    assert not campaign.retry_authorized_provider_failure(item, status, None)
    assert calls == [1, 2, 3]
