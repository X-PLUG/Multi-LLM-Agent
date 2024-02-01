__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup


source activate
conda activate /nas-wulanchabu/miniconda3/envs/weizhou_tool

cd /nas-wulanchabu/shenweizhou.swz/tool_learning/ToolBench-new

export PYTHONPATH=./
python toolbench/inference/LLM/collab_agent_model.py 