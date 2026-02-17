# Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities

A challenging aggregation benchmark for long-context models; full code and eval scripts release coming soon! See the [paper](https://arxiv.org/abs/2511.02817) for details.


## Running Oolong

The example inference script assumes you use LiteLLM as a wrapper around your API of choice. 

set up:
```sh
pip install -r requirements.txt
```

inference:
```bash
export MODEL="gpt-oss-20b"
export PORT=8000
# start vllm server at port $PORT serving the model
# vllm serve ${HF_MODEL_ID_OR_LOCAL_PATH} --port ${PORT} --served-model-name ${MODEL}

# read the Oolong paper to understand the flags
# `min_context_len` and `max_context_len` set the context length to evaluate on

python src/eval/eval_script_batched.py \
    --model "hosted_vllm/${MODEL}" --reasoning_level low \
    --dataset synth --split test \
    --min_context_len 65536 --max_context_len 65536 \
    --base_url "http://0.0.0.0:${PORT}/v1"
```

This will run inference with a fixed batch size 1.

You can set maximum and minimum input example lengths; Oolong-real will also attempt to infer the maximum input length from the model provided.


## BibTeX

```
@misc{bertsch2025oolongevaluatinglongcontext,
      title={Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities}, 
      author={Amanda Bertsch and Adithya Pratapa and Teruko Mitamura and Graham Neubig and Matthew R. Gormley},
      year={2025},
      eprint={2511.02817},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.02817}, 
}
```
