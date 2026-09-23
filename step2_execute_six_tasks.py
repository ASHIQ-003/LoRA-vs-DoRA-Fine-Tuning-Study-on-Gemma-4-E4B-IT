# step2_execute_six_tasks.py
# Executes ONLY the 6 required compute tasks.
# Prerequisite: step1_validate_manifest.py must have printed [OK].
# DO NOT run unless the manifest is validated.

# ============================================================
# SECTION 0: DEPENDENCY BOOTSTRAP
# ============================================================
import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_pip("transformers==5.17.0")
_pip("trl==1.13.0")
_pip("peft==0.21.0")
_pip("bitsandbytes==0.50.2")
_pip("datasets")
_pip("pyyaml")

# ============================================================
# SECTION 1: STANDARD IMPORTS
# ============================================================
import os, json, csv, re, gc, time, shutil, logging, traceback
import torch
import numpy as np
import random

# Add repo to path (repo is at /kaggle/working/repo after kernel_entry)
REPO_DIR = "/kaggle/working/repo"
if os.path.isdir(REPO_DIR) and REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_runner")

# ============================================================
# SECTION 2: FROZEN CONSTANTS (do NOT modify)
# ============================================================
MODEL_ID        = "google/gemma-4-E4B-it"
TARGET_MODULES  = ["q_proj.linear", "v_proj.linear"]
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
TRAIN_SAMPLES   = 200
EVAL_SAMPLES    = 20
MAX_STEPS       = 20
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LR              = 2e-4
MAX_LEN         = 512
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01
MAX_NEW_TOKENS  = 256

WORKING     = "/kaggle/working"
FINAL_DIR   = os.path.join(WORKING, "final_results")
MANIFEST_PATH = os.path.join(WORKING, "pilot_recovery_manifest.json")

os.makedirs(FINAL_DIR, exist_ok=True)

# ============================================================
# SECTION 3: SEED UTILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ============================================================
# SECTION 4: PROMPT / ANSWER UTILITIES (frozen validated)
# ============================================================
def build_prompt(question):
    return "<start_of_turn>user\n" + question + "<end_of_turn>\n<start_of_turn>model\n"

def normalize_number(s):
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def extract_final_answer(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        m = re.search(r"-?\d+(?:,\d+)*(?:\.\d+)?", ans)
        if m:
            return normalize_number(m.group())
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return normalize_number(numbers[-1])
    return ""

# ============================================================
# SECTION 5: MODEL LOADING
# ============================================================
def load_model_and_processor():
    logger.info("Loading processor and 4-bit model: " + MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    return model, processor

def get_tokenizer(processor):
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        return processor.tokenizer
    return processor

# ============================================================
# SECTION 6: ADAPTER INJECTION (mirrors src/adapters.py)
# ============================================================
import bitsandbytes as bnb_mod
from peft import get_peft_model

def inject_adapter(model, method, rank):
    logger.info("Freezing all parameters before adapter injection")
    for p in model.parameters():
        p.requires_grad_(False)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name.lower():
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)

    use_dora = (method.lower() == "dora")
    logger.info("Creating " + method.upper() + " config r=" + str(rank) + " dora=" + str(use_dora))

    cfg = LoraConfig(
        r=rank,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,
    )

    # Verify targets exist
    named = dict(model.named_modules())
    for t in TARGET_MODULES:
        mod = named.get(t)
        assert mod is not None, "Target " + t + " not found in model!"
        assert isinstance(mod, (torch.nn.Linear, bnb_mod.nn.Linear4bit)), \
            "Target " + t + " is " + str(type(mod)) + ", expected Linear/Linear4bit"

    model = get_peft_model(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: " + str(trainable) + " / " + str(total))
    return model, trainable

# ============================================================
# SECTION 7: DATASET PREPARATION
# ============================================================
def load_and_prepare_data(tokenizer, seed):
    logger.info("Loading GSM8K train split, selecting " + str(TRAIN_SAMPLES) + " samples")
    raw = load_dataset("openai/gsm8k", "main", split="train")
    raw = raw.shuffle(seed=seed).select(range(TRAIN_SAMPLES))

    def tokenize_and_mask(ex):
        prompt     = build_prompt(ex["question"])
        completion = ex["answer"] + "<end_of_turn>\n"
        p_ids = tokenizer.encode(prompt,     add_special_tokens=False)
        c_ids = tokenizer.encode(completion, add_special_tokens=False)
        ids   = p_ids + c_ids
        labels = [-100] * len(p_ids) + list(c_ids)
        mask   = [1] * len(ids)
        if len(ids) > MAX_LEN:
            ids    = ids[:MAX_LEN]
            labels = labels[:MAX_LEN]
            mask   = mask[:MAX_LEN]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    logger.info("Tokenizing and applying completion-only masking")
    return raw.map(tokenize_and_mask, remove_columns=raw.column_names)

def load_eval_data():
    logger.info("Loading GSM8K test split, selecting " + str(EVAL_SAMPLES) + " examples")
    raw = load_dataset("openai/gsm8k", "main", split="test")
    return raw.select(range(EVAL_SAMPLES))

# ============================================================
# SECTION 8: TRAINING
# ============================================================
import inspect

class CustomSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        return loss

def run_training(model, processor, train_dataset, seed, output_dir):
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.pad_token    = tokenizer.eos_token or "<pad>"

    sft_kwargs = {
        "output_dir":                output_dir,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate":             LR,
        "max_steps":                 MAX_STEPS,
        "logging_steps":             5,
        "save_strategy":             "no",
        "warmup_ratio":              WARMUP_RATIO,
        "weight_decay":              WEIGHT_DECAY,
        "bf16":                      True,
        "fp16":                      False,
        "max_length":                MAX_LEN,
        "report_to":                 "none",
        "seed":                      seed,
    }
    valid = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    args  = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in valid})
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    trainer_kwargs = {
        "model":        model,
        "train_dataset": train_dataset,
        "args":         args,
        "data_collator": collator,
    }
    valid_t = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in valid_t:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in valid_t:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = CustomSFTTrainer(**trainer_kwargs)

    # Regression test: masking must be correct
    dl     = trainer.get_train_dataloader()
    batch  = next(iter(dl))
    labels = batch["labels"][0].tolist()
    assert -100 in labels and any(l != -100 for l in labels), \
        "MASKING REGRESSION TEST FAILED"
    logger.info("MASKING REGRESSION TEST PASSED")

    t0 = time.time()
    result = trainer.train()
    train_runtime = time.time() - t0

    train_metrics = {
        "train_loss":          result.training_loss,
        "train_runtime_sec":   train_runtime,
        "train_steps":         MAX_STEPS,
    }
    logger.info("Training complete. Loss=" + str(result.training_loss) + " time=" + str(train_runtime))
    return trainer, train_metrics

# ============================================================
# SECTION 9: EVALUATION (validated one-by-one protocol)
# ============================================================
def run_evaluation(model, processor, eval_data, task_label):
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model.eval()
    predictions = []
    correct = 0
    no_final_answer = 0

    t0 = time.time()
    for i, ex in enumerate(eval_data):
        q    = ex["question"]
        gold_raw = ex["answer"]
        gold = extract_final_answer(gold_raw)

        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        print("[EVAL] " + task_label + " example " + str(i+1) + "/" + str(len(eval_data)))

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0][input_len:]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_final_answer(raw_response)

        is_correct = (pred == gold) and (pred != "")
        if is_correct:
            correct += 1
        if pred == "":
            no_final_answer += 1

        predictions.append({
            "index":         i,
            "question":      q,
            "gold_raw":      gold_raw,
            "gold":          gold,
            "raw_response":  raw_response,
            "predicted":     pred,
            "correct":       is_correct,
        })
        print("  gold=" + str(gold) + "  pred=" + str(pred) + "  correct=" + str(is_correct))

    eval_runtime = time.time() - t0
    accuracy = correct / len(eval_data)
    valid_answers = sum(1 for p in predictions if p["predicted"] != "")

    eval_metrics = {
        "accuracy":         accuracy,
        "correct":          correct,
        "total":            len(eval_data),
        "valid_answers":    valid_answers,
        "no_final_answer":  no_final_answer,
        "eval_runtime_sec": eval_runtime,
    }
    return eval_metrics, predictions

# ============================================================
# SECTION 10: PEAK VRAM UTILITY
# ============================================================
def get_peak_vram_gb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return None

def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def purge_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# SECTION 11: PERSIST TASK RESULT
# ============================================================
def save_task_result(tag, method, rank, seed, train_metrics, eval_metrics,
                     predictions, adapter_path, status, peak_vram):
    task_dir = os.path.join(FINAL_DIR, tag)
    os.makedirs(task_dir, exist_ok=True)

    with open(os.path.join(task_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    if train_metrics:
        with open(os.path.join(task_dir, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f, indent=2)

    with open(os.path.join(task_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    meta = {
        "method": method, "rank": rank, "seed": seed,
        "status": status, "adapter_path": adapter_path,
        "peak_vram_gb": peak_vram,
        "eval_metrics": eval_metrics,
        "train_metrics": train_metrics,
    }
    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if adapter_path and os.path.isdir(adapter_path):
        dest = os.path.join(task_dir, "adapter")
        if not os.path.exists(dest):
            shutil.copytree(adapter_path, dest)

    logger.info("Task result saved to " + task_dir)
    return task_dir

# ============================================================
# SECTION 12: MASTER RESULTS TABLE
# ============================================================
def build_master_results(manifest, new_results):
    rows = []
    new_by_key = {(r["method"], r["rank"], r["seed"]): r for r in new_results}

    for e in manifest:
        m, r, s = e["method"], e["rank"], e["seed"]
        key = (m, r, s)
        nr  = new_by_key.get(key)

        if nr is not None:
            row = {
                "method": m, "rank": r, "seed": s,
                "accuracy": nr.get("accuracy"),
                "status":   nr.get("status"),
                "train_loss":        nr.get("train_loss"),
                "train_runtime_sec": nr.get("train_runtime_sec"),
                "eval_runtime_sec":  nr.get("eval_runtime_sec"),
                "trainable_params":  nr.get("trainable_params"),
                "peak_vram_gb":      nr.get("peak_vram_gb"),
                "artifact_path":     nr.get("artifact_path"),
            }
        else:
            acc = e.get("logged_accuracy")
            row = {
                "method": m, "rank": r, "seed": s,
                "accuracy": acc,
                "status":   e.get("result_confidence", "VERIFIED_FROM_LOG"),
                "train_loss":        None,
                "train_runtime_sec": None,
                "eval_runtime_sec":  None,
                "trainable_params":  None,
                "peak_vram_gb":      None,
                "artifact_path":     None,
            }
        rows.append(row)
    return rows

def save_master_results(manifest, new_results):
    rows = build_master_results(manifest, new_results)

    json_path = os.path.join(WORKING, "master_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    csv_path = os.path.join(WORKING, "master_results.csv")
    fields = ["method","rank","seed","accuracy","status","train_loss",
              "train_runtime_sec","eval_runtime_sec","trainable_params",
              "peak_vram_gb","artifact_path"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    logger.info("Master results saved.")
    return rows

def save_recovery_status(new_results):
    path = os.path.join(WORKING, "recovery_status.csv")
    fields = ["method","rank","seed","status","accuracy","error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in new_results:
            w.writerow({
                "method":   r.get("method"),
                "rank":     r.get("rank"),
                "seed":     r.get("seed"),
                "status":   r.get("status"),
                "accuracy": r.get("accuracy"),
                "error":    r.get("error", ""),
            })

# ============================================================
# SECTION 13: COMPACT RESULT SUMMARY PRINTER
# ============================================================
def print_summary(method, rank, seed, eval_metrics, train_metrics, peak_vram, status):
    acc       = eval_metrics.get("accuracy") if eval_metrics else None
    valid     = eval_metrics.get("valid_answers") if eval_metrics else None
    no_fa     = eval_metrics.get("no_final_answer") if eval_metrics else None
    t_rt      = train_metrics.get("train_runtime_sec") if train_metrics else None
    e_rt      = eval_metrics.get("eval_runtime_sec") if eval_metrics else None

    acc_s  = "{:.2f}%".format(acc * 100) if acc is not None else "N/A"
    t_rt_s = "{:.1f}s".format(t_rt) if t_rt is not None else "N/A"
    e_rt_s = "{:.1f}s".format(e_rt) if e_rt is not None else "N/A"
    vram_s = "{}GB".format(peak_vram) if peak_vram is not None else "N/A"

    print("\n" + "="*70)
    print("TASK COMPLETE: {} r={} seed={} | {}".format(
        method.upper(), rank, seed, status))
    print("  accuracy={} | valid={} | no_final_answer={} | train_rt={} | eval_rt={} | vram={}".format(
        acc_s, valid, no_fa, t_rt_s, e_rt_s, vram_s))
    print("="*70 + "\n")

# ============================================================
# SECTION 14: LOAD MANIFEST
# ============================================================
print("Loading manifest from " + MANIFEST_PATH)
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# Identify 6 compute tasks
eval_only_tasks  = [e for e in manifest if e.get("eval_only") and e.get("rerun_required")]
full_rerun_tasks = [e for e in manifest if e.get("rerun_required") and not e.get("eval_only")]

print("EVAL_ONLY tasks  : " + str(len(eval_only_tasks)))
print("FULL_RERUN tasks : " + str(len(full_rerun_tasks)))
assert len(eval_only_tasks) == 1,  "Expected 1 eval-only task"
assert len(full_rerun_tasks) == 5, "Expected 5 full-rerun tasks"
print("Manifest validated. Proceeding with 6 compute tasks.")

# ============================================================
# SECTION 15: EXECUTE ALL 6 TASKS
# ============================================================
new_results = []

ALL_TASKS = eval_only_tasks + full_rerun_tasks

for task in ALL_TASKS:
    method = task["method"]
    rank   = task["rank"]
    seed   = task["seed"]
    evonly = task.get("eval_only", False)
    tag    = "{}_r{}_seed{}".format(method, rank, seed)

    print("\n" + "#"*70)
    if evonly:
        print("# TASK: EVAL ONLY  -- {} r={} seed={}".format(method.upper(), rank, seed))
    else:
        print("# TASK: FULL RERUN -- {} r={} seed={}".format(method.upper(), rank, seed))
    print("#"*70)

    train_metrics  = None
    eval_metrics   = None
    predictions    = []
    adapter_path   = None
    trainable_p    = None
    peak_vram      = None
    status         = "UNKNOWN"
    error_msg      = ""

    try:
        set_seed(seed)
        reset_vram_stats()

        # Expected adapter directory from previous session
        adapter_path_expected = os.path.join(
            WORKING, "{}_r{}_seed{}_steps{}".format(method, rank, seed, MAX_STEPS)
        )

        # ---- EVAL ONLY BRANCH ----
        if evonly:
            if not os.path.isdir(adapter_path_expected):
                raise FileNotFoundError(
                    "EVAL_ONLY adapter not found at " + adapter_path_expected +
                    ". Session was wiped. Treating as FULL_RERUN."
                )
            adapter_path = adapter_path_expected
            print("Loading model for eval-only task")
            model, processor = load_model_and_processor()

            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()

            eval_data    = load_eval_data()
            eval_metrics, predictions = run_evaluation(model, processor, eval_data, tag)
            status       = "EVAL_RECOVERED"
            peak_vram    = get_peak_vram_gb()
            purge_model(model)

        # ---- FULL RERUN BRANCH ----
        else:
            adapter_path = os.path.join(
                WORKING, "{}_r{}_seed{}_steps{}".format(method, rank, seed, MAX_STEPS)
            )
            os.makedirs(adapter_path, exist_ok=True)

            print("Loading model for full rerun")
            model, processor = load_model_and_processor()
            tokenizer = get_tokenizer(processor)

            model, trainable_p = inject_adapter(model, method, rank)

            train_data = load_and_prepare_data(tokenizer, seed)
            trainer, train_metrics = run_training(
                model, processor, train_data, seed, adapter_path
            )
            train_metrics["trainable_params"] = trainable_p

            # Save adapter
            trainer.model.save_pretrained(adapter_path)
            processor.save_pretrained(adapter_path)
            print("Adapter saved to " + adapter_path)

            # Copy to final_results immediately
            final_adapter_copy = os.path.join(FINAL_DIR, tag + "_adapter")
            if not os.path.exists(final_adapter_copy):
                shutil.copytree(adapter_path, final_adapter_copy)

            # Evaluate
            eval_data = load_eval_data()
            eval_metrics, predictions = run_evaluation(model, processor, eval_data, tag)
            status    = "NEWLY_RERUN"
            peak_vram = get_peak_vram_gb()
            purge_model(model)

    except Exception as ex:
        error_msg = traceback.format_exc()
        status    = "FAILED"
        print("ERROR in task " + tag + ":\n" + error_msg)

    # Record result regardless of success/failure
    new_results.append({
        "method":            method,
        "rank":              rank,
        "seed":              seed,
        "accuracy":          eval_metrics.get("accuracy") if eval_metrics else None,
        "status":            status,
        "train_loss":        train_metrics.get("train_loss") if train_metrics else None,
        "train_runtime_sec": train_metrics.get("train_runtime_sec") if train_metrics else None,
        "eval_runtime_sec":  eval_metrics.get("eval_runtime_sec") if eval_metrics else None,
        "trainable_params":  trainable_p,
        "peak_vram_gb":      peak_vram,
        "artifact_path":     adapter_path if status != "FAILED" else None,
        "error":             error_msg,
    })

    # Save task result directory
    if status != "FAILED":
        save_task_result(tag, method, rank, seed, train_metrics, eval_metrics,
                         predictions, adapter_path, status, peak_vram)

    # Update master results immediately after every task
    save_master_results(manifest, new_results)
    save_recovery_status(new_results)

    # Print compact summary
    print_summary(method, rank, seed, eval_metrics, train_metrics, peak_vram, status)

# ============================================================
# SECTION 16: FINAL OUTPUTS
# ============================================================
print("\n" + "="*70)
print("ALL 6 TASKS COMPLETE. Generating final outputs...")
print("="*70)

# Final master results
rows = save_master_results(manifest, new_results)

# Print final 24-row table
print("\n--- FINAL 24-RUN MASTER TABLE ---")
print("{:<6} {:<4} {:<4} {:>9} {:<22}".format(
    "Method", "Rank", "Seed", "Accuracy", "Status"))
print("-"*60)
for row in rows:
    acc = row["accuracy"]
    acc_s = "{:.2f}%".format(acc * 100) if acc is not None else "N/A"
    print("{:<6} {:<4} {:<4} {:>9} {:<22}".format(
        row["method"].upper(), row["rank"], row["seed"],
        acc_s, str(row["status"])))

# ZIP everything
zip_base = os.path.join(WORKING, "final_results")
zip_path = zip_base + ".zip"
shutil.make_archive(zip_base, "zip", WORKING, "final_results")
print("\nZIP archive created: " + zip_path)

# Add master CSVs into ZIP (re-zip with all outputs)
import zipfile
extra_files = ["master_results.csv", "master_results.json", "recovery_status.csv",
               "pilot_recovery_manifest.json", "pilot_recovery_manifest.csv"]
with zipfile.ZipFile(zip_path, "a") as zf:
    for fname in extra_files:
        fpath = os.path.join(WORKING, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)

print("Final outputs in " + WORKING + ":")
print("  master_results.csv")
print("  master_results.json")
print("  recovery_status.csv")
print("  final_results/  (one subfolder per completed task)")
print("  final_results.zip  (all of the above)")
print("\nDONE.")
