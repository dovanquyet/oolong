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
    --dataset synth --split validation \
    --min_context_len 16384 --max_context_len 16384 \
    --base_url "http://0.0.0.0:${PORT}/v1" --batch_size 1
```

This will run inference with batch size 1; you can set a batch size with `--batch_size`, or pass `--batch_by_context_window` to pass a single example from each context window to seed the cache and then the remainder of examples from that window in a single batch. 

You can set maximum and minimum input example lengths; Oolong-real will also attempt to infer the maximum input length from the model provided.

## More soon!

Release status:
- [x] Output scoring scripts for both splits
- [x] API inference script
- [ ] Oolong-synth construction code
- [ ] Validated splits of each Oolong-synth source dataset 
- [ ] Oolong-real construction code
- [ ] Full output sets for models from the paper
- [ ] Analysis scripts


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
