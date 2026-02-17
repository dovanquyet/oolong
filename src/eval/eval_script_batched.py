import os, time
import jsonlines
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset
import litellm
# litellm._turn_on_debug()

from eval_helpers import dnd_process_response, synth_process_response
from utils import compute_context_lengths



def llm_call(args, datapoint, split_to_use, response_processor, api_kwargs, 
             cache_control={"cache_control": {"type": "ephemeral"}}, max_retries=3):
    """
    Make an LLM call and interpret the output with automatic retry on parsing errors.
    
    Args:
        args: Arguments including base_url and model
        datapoint: The data point to process
        split_to_use: The key for the context text in the datapoint
        response_processor: Function to process the response
        api_kwargs: Additional API arguments
        cache_control: Dictionary to control caching behavior for the LLM call (default: ephemeral)
        max_retries: Maximum number of retries on error
    
    Returns:
        Processed output dictionary
    """
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                api_key=os.environ.get("LITELLM_API_KEY"),
                base_url=args.base_url,
                tools=[],
                model=args.model,
                # api_version="2024-12-01",
                # extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
                messages=[
                    {"role": "system", "content": [
                        {"type": "text", "text": "You are a helpful assistant."},
                        {"type": "text", "text": f"{datapoint[split_to_use]}", **cache_control},
                    ]},
                    {"role": "user", "content": f"{datapoint['question']}"},
                ],
                **api_kwargs,
            )
            
            # Process the response
            output = response["choices"][0]["message"]["content"]
            if output is None:
                if response["choices"][0]["finish_reason"] == "content_filter":
                    output = "CONTENT_FILTERED"
                    print("WARNING: CONTENT FILTERED")
                else:
                    raise ValueError("empty output!")
            
            return response_processor(datapoint, output, args.model)
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    "Error during LLM call for datapoint "
                    f"{datapoint['id']} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(3 ** (attempt+1))  # Exponential backoff
            else:
                print(f"Final error after {max_retries} attempts for datapoint {datapoint['id']}: {e}")
                return {
                    "id": datapoint["id"],
                    "context_window_id": datapoint["context_window_id"],
                    "dataset": datapoint["dataset"],
                    "model": args.model,
                    "attempted_parse": "ERROR",
                    "parse_confidence": "ERROR",
                    "full_answer": "ERROR",
                    "score": 0,
                    "context_len": datapoint["context_len"],
                    "task_group": datapoint["task_group"],
                    "task": datapoint["task"],
                    "answer_type": datapoint["answer_type"],
                    "answer": str(datapoint["answer"].strip("[").strip("]")),
                }


def launch(
    dataset,
    split,
    labels,
    max_context_len,
    min_context_len,
    model,
    reasoning_level,
    args
):
    # download, filter, and sort data
    if dataset == "synth":
        data = load_dataset("oolongbench/oolong-synth")[split]
        process_response = synth_process_response
    else:
        # use 'toy_dnd' config to try out the DnD dataset
        data = load_dataset("oolongbench/oolong-real", "dnd")[split]
        process_response = dnd_process_response
        # we compute token counts based on the model's tokenizer
        data = compute_context_lengths(data, model)

    data = data.filter(lambda x: x["context_len"] <= max_context_len)
    data = data.filter(lambda x: x["context_len"] >= min_context_len)
    print(
        f"Evaluating {len(data)} examples for model {model} "
        f"with context lengths between {min_context_len} and {max_context_len}..."
    )
    data = data.sort("context_window_id") # sort by context window ID to enable caching

    # config i/o
    results_dir = "results"
    safemodelname = model.split("/")[-1]  # +"-labels"
    split_to_use = "context_window_text"
    api_kwargs = {}

    if labels:
        safemodelname += "-labels"
        split_to_use = "context_window_text_with_labels"

    if reasoning_level != "":
        safemodelname += f"-{reasoning_level}"
        api_kwargs["reasoning_effort"] = reasoning_level
        api_kwargs["extra_body"] = {"allowed_openai_params": ["reasoning_effort"]}

    Path(f"{results_dir}/{dataset}/{safemodelname}").mkdir(parents=True, exist_ok=True)
    full_results_path = f"{results_dir}/{dataset}/{safemodelname}/full_output.jsonl"

    # init stats
    output_counter = 0
    correct = 0
    all_considered_ids = list(data["id"])
    total_count = len(all_considered_ids)

    # potentially init from prior partial run
    if os.path.exists(full_results_path):
        ids_to_skip = []
        correct = 0
        output_counter = 0

        with jsonlines.open(full_results_path, "r") as f:
            for obj in f:
                ids_to_skip.append(obj["id"])
                if obj["id"] in all_considered_ids:
                    correct += obj["score"]
                    output_counter += 1

        data = data.filter(lambda x: x["id"] not in ids_to_skip)
        print(
            "Caution: filtered out completed examples; "
            f"{len(data)} examples left to run..."
        )

    else:
        with jsonlines.open(full_results_path, "w") as f:
            pass  # init file

    fout_full_output = jsonlines.open(full_results_path, "a")
    for datapoint in tqdm(data, desc="Evaluating examples", total=len(data)):
        try:
            this_output = llm_call(
                args, datapoint, split_to_use, process_response, api_kwargs,
            )
            correct += this_output["score"]
            output_counter += 1
            fout_full_output.write(this_output)
            print(f"Score so far: {correct / output_counter:.4f} ({output_counter} examples)")

        except Exception as e:
            print(f"Error on datapoint {datapoint['id']}, which is item {output_counter}: {e}")

    with open(f"{results_dir}/{dataset}/{safemodelname}/overall.txt", "w") as f:
        summary = f"Overall score for {model} on {total_count} examples:" + \
                  f"{correct}/{total_count} = {correct / total_count}"
        f.write(summary)
        print(summary)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument(
        "--dataset",
        default="synth",
        choices=["synth", "real"],
        help="Dataset to use (default: synth)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to evaluate the model on",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        default=False,
        help="Enable labels (default: False)",
    )
    parser.add_argument(
        "--max_context_len",
        default=131072,
        type=int,
        help="max context length to include",
    )
    parser.add_argument(
        "--min_context_len",
        default=8192,
        type=int,
        help="min context length to include",
    )

    # model config
    parser.add_argument(
        "--model", 
        required=True,
        type=str,
        help="Model name with a router prefix if necessary"
        "(e.g. 'litellm_proxy/gpt-4o', 'gemini/gemini-3-pro-preview', 'hosted_vllm/gpt-oss-20b')"
    )
    parser.add_argument(
        "--reasoning_level",
        default="",
        choices=["", "low", "medium", "high", "minimal", "none", "disabled"],
        help="Reasoning level (default: empty string)",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        type=str,
        help="a base URL for a hosted litellm instance, if necessary",
    )

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    launch(
        args.dataset,
        args.split,
        args.labels,
        args.max_context_len,
        args.min_context_len,
        args.model,
        args.reasoning_level,
        args
    )
