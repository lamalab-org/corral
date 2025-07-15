## ML single experiments

export CORRAL_WORK_DIR="/Users/n0w0f/CORRAL_WORK_DIR/tool_ablation/react/ml_single/claude_comprehensive"
python -m env /Users/n0w0f/git/n0w0f/mat-agent-bench/tasks/ml/config/dataset.json
cd /Users/n0w0f/git/n0w0f/mat-agent-bench/reports/gpt4o/ml_single
python run_react_agent.py | tee gpt-react-ml_single-comprehensive_verbosity.log

export CORRAL_WORK_DIR="/Users/n0w0f/CORRAL_WORK_DIR/tool_ablation/tool/ml_single/claude_comprehensive"
python -m env1 /Users/n0w0f/git/n0w0f/mat-agent-bench/tasks/ml/config/dataset.json
cd /Users/n0w0f/git/n0w0f/mat-agent-bench/reports/gpt4o/ml_single
python run_toolcalling_agent.py | tee gpt-toolcalling-ml_single-comprehensive_verbosity.log

export CORRAL_WORK_DIR="/Users/n0w0f/CORRAL_WORK_DIR/tool_ablation/react/ml_single/gpt_comprehensive"
python -m env2 /Users/n0w0f/git/n0w0f/mat-agent-bench/tasks/ml/config/dataset.json
cd /Users/n0w0f/git/n0w0f/mat-agent-bench/reports/claude/ml_single/
python run_react_agent.py | tee claude-react-ml_single-comprehensive_verbosity.log

export CORRAL_WORK_DIR="/Users/n0w0f/CORRAL_WORK_DIR/tool_ablation/tool/ml_single/gpt_comprehensive"
python -m env3 /Users/n0w0f/git/n0w0f/mat-agent-bench/tasks/ml/config/dataset.json

python run_toolcalling_agent.py | tee claude-toolcalling-ml_single-comprehensive_verbosity.log
