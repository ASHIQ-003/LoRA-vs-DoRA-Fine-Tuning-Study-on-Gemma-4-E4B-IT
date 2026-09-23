# step2_main.py  — paste this as Cell 2 AFTER running pip installs in Cell 1
# No pip bootstrap here. Deps already installed.

import os, json, csv, re, gc, time, shutil, logging, traceback
import sys, random, inspect, zipfile
import torch
import numpy as np

# Add repo to sys.path
REPO_DIR = "/kaggle/working/repo"
if os.path.isdir(REPO_DIR) and REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import bitsandbytes as bnb_mod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_runner")

# ── FROZEN CONSTANTS ──────────────────────────────────────────────────────────
MODEL_ID       = "google/gemma-4-E4B-it"
TARGET_MODULES = ["q_proj.linear", "v_proj.linear"]
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
TRAIN_SAMPLES  = 200
EVAL_SAMPLES   = 20
MAX_STEPS      = 20
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 2e-4
MAX_LEN        = 512
WARMUP_RATIO   = 0.1
WEIGHT_DECAY   = 0.01
MAX_NEW_TOKENS = 256

WORKING      = "/kaggle/working"
FINAL_DIR    = os.path.join(WORKING, "final_results")
MANIFEST_PATH = os.path.join(WORKING, "pilot_recovery_manifest.json")
os.makedirs(FINAL_DIR, exist_ok=True)

# ── SEED ──────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ── PROMPT / ANSWER ───────────────────────────────────────────────────────────
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

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_model_and_processor():
    print("Loading processor and 4-bit model: " + MODEL_ID)
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

# ── ADAPTER INJECTION ─────────────────────────────────────────────────────────
def inject_adapter(model, method, rank):
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name.lower():
            param.data = param.data.to(torch.float32)
            param.requires_grad_(False)

    use_dora = (method.lower() == "dora")
    print("Injecting " + method.upper() + " adapter r=" + str(rank) + " use_dora=" + str(use_dora))
    cfg = LoraConfig(
        r=rank, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
        use_dora=use_dora,
    )
    named = dict(model.named_modules())
    for t in TARGET_MODULES:
        mod = named.get(t)
        assert mod is not None, "Target " + t + " not found!"
        assert isinstance(mod, (torch.nn.Linear, bnb_mod.nn.Linear4bit)), \
            "Target " + t + " wrong type: " + str(type(mod))

    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print("Trainable params: " + str(trainable) + " / " + str(total))
    return model, trainable

# ── DATASET ───────────────────────────────────────────────────────────────────
def load_and_prepare_train_data(tokenizer, seed):
    print("Loading GSM8K train, selecting " + str(TRAIN_SAMPLES) + " samples (seed=" + str(seed) + ")")
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

    return raw.map(tokenize_and_mask, remove_columns=raw.column_names)

def load_eval_data():
    print("Loading GSM8K test, selecting " + str(EVAL_SAMPLES) + " examples")
    raw = load_dataset("openai/gsm8k", "main", split="test")
    return raw.select(range(EVAL_SAMPLES))

# ── TRAINING ──────────────────────────────────────────────────────────────────
class CustomSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

def run_training(model, processor, train_dataset, seed, output_dir):
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        tokenizer.pad_token    = tokenizer.eos_token or "<pad>"

    sft_kwargs = {
        "output_dir":                  output_dir,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate":               LR,
        "max_steps":                   MAX_STEPS,
        "logging_steps":               5,
        "save_strategy":               "no",
        "warmup_ratio":                WARMUP_RATIO,
        "weight_decay":                WEIGHT_DECAY,
        "bf16":                        True,
        "fp16":                        False,
        "max_length":                  MAX_LEN,
        "report_to":                   "none",
        "seed":                        seed,
    }
    valid = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    args  = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in valid})
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    trainer_kwargs = {
        "model":         model,
        "train_dataset": train_dataset,
        "args":          args,
        "data_collator": collator,
    }
    valid_t = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in valid_t:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in valid_t:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = CustomSFTTrainer(**trainer_kwargs)

    dl     = trainer.get_train_dataloader()
    batch  = next(iter(dl))
    labels = batch["labels"][0].tolist()
    assert -100 in labels and any(l != -100 for l in labels), \
        "MASKING REGRESSION FAILED"
    print("Masking regression test PASSED")

    t0 = time.time()
    result = trainer.train()
    train_runtime = time.time() - t0
    return trainer, {
        "train_loss":        result.training_loss,
        "train_runtime_sec": train_runtime,
        "train_steps":       MAX_STEPS,
    }

# ── EVALUATION ────────────────────────────────────────────────────────────────
def run_evaluation(model, processor, eval_data, label):
    tokenizer = get_tokenizer(processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    model.eval()

    predictions = []
    correct = no_fa = 0
    t0 = time.time()

    for i, ex in enumerate(eval_data):
        q    = ex["question"]
        gold = extract_final_answer(ex["answer"])
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        print("[EVAL] " + label + " " + str(i+1) + "/" + str(len(eval_data)) +
              "  gold=" + str(gold))
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
        raw  = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        pred = extract_final_answer(raw)
        ok   = (pred == gold) and (pred != "")
        if ok:  correct += 1
        if pred == "": no_fa += 1
        predictions.append({
            "index": i, "question": q,
            "gold": gold, "raw_response": raw,
            "predicted": pred, "correct": ok,
        })
        print("           pred=" + str(pred) + "  correct=" + str(ok))

    accuracy = correct / len(eval_data)
    return {
        "accuracy":         accuracy,
        "correct":          correct,
        "total":            len(eval_data),
        "valid_answers":    sum(1 for p in predictions if p["predicted"] != ""),
        "no_final_answer":  no_fa,
        "eval_runtime_sec": time.time() - t0,
    }, predictions

# ── VRAM ──────────────────────────────────────────────────────────────────────
def get_peak_vram():
    return round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None

def reset_vram():
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

def purge(model):
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── PERSISTENCE ───────────────────────────────────────────────────────────────
def save_task(tag, method, rank, seed, tm, em, preds, adapter_path, status, vram):
    d = os.path.join(FINAL_DIR, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "eval_metrics.json"),  "w") as f: json.dump(em or {}, f, indent=2)
    with open(os.path.join(d, "predictions.json"),   "w") as f: json.dump(preds, f, indent=2)
    with open(os.path.join(d, "train_metrics.json"), "w") as f: json.dump(tm or {}, f, indent=2)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump({"method":method,"rank":rank,"seed":seed,"status":status,
                   "peak_vram_gb":vram,"eval_metrics":em,"train_metrics":tm}, f, indent=2)
    if adapter_path and os.path.isdir(adapter_path):
        dest = os.path.join(d, "adapter")
        if not os.path.exists(dest):
            shutil.copytree(adapter_path, dest)
    print("Saved task result to " + d)

def save_master(manifest, new_results):
    new_by_key = {(r["method"], r["rank"], r["seed"]): r for r in new_results}
    rows = []
    for e in manifest:
        m, r, s = e["method"], e["rank"], e["seed"]
        nr = new_by_key.get((m, r, s))
        if nr:
            rows.append({
                "method":m,"rank":r,"seed":s,
                "accuracy":nr.get("accuracy"),"status":nr.get("status"),
                "train_loss":nr.get("train_loss"),
                "train_runtime_sec":nr.get("train_runtime_sec"),
                "eval_runtime_sec":nr.get("eval_runtime_sec"),
                "trainable_params":nr.get("trainable_params"),
                "peak_vram_gb":nr.get("peak_vram_gb"),
                "artifact_path":nr.get("artifact_path"),
            })
        else:
            rows.append({
                "method":m,"rank":r,"seed":s,
                "accuracy":e.get("logged_accuracy"),
                "status":e.get("result_confidence","VERIFIED_FROM_LOG"),
                "train_loss":None,"train_runtime_sec":None,
                "eval_runtime_sec":None,"trainable_params":None,
                "peak_vram_gb":None,"artifact_path":None,
            })
    fields = ["method","rank","seed","accuracy","status","train_loss",
              "train_runtime_sec","eval_runtime_sec","trainable_params",
              "peak_vram_gb","artifact_path"]
    with open(os.path.join(WORKING,"master_results.json"),"w") as f: json.dump(rows,f,indent=2)
    with open(os.path.join(WORKING,"master_results.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(os.path.join(WORKING,"recovery_status.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","rank","seed","status","accuracy","error"])
        w.writeheader()
        for r in new_results:
            w.writerow({"method":r["method"],"rank":r["rank"],"seed":r["seed"],
                        "status":r["status"],"accuracy":r.get("accuracy"),"error":r.get("error","")})
    print("Master results updated.")
    return rows

def print_summary(method, rank, seed, em, tm, vram, status):
    acc  = "{:.2f}%".format(em["accuracy"]*100) if em else "N/A"
    val  = em.get("valid_answers")  if em else "N/A"
    nofa = em.get("no_final_answer") if em else "N/A"
    trt  = "{:.1f}s".format(tm["train_runtime_sec"]) if tm else "N/A"
    ert  = "{:.1f}s".format(em["eval_runtime_sec"])  if em else "N/A"
    vr   = str(vram)+"GB" if vram else "N/A"
    print("\n" + "="*65)
    print("DONE: {} r={} seed={} | {}".format(method.upper(),rank,seed,status))
    print("  acc={}  valid={}  no_fa={}  train={}  eval={}  vram={}".format(
        acc,val,nofa,trt,ert,vr))
    print("="*65 + "\n")

# ── LOAD & VALIDATE MANIFEST ──────────────────────────────────────────────────
print("Reading manifest: " + MANIFEST_PATH)
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

eval_only_tasks  = [e for e in manifest if e.get("eval_only") and e.get("rerun_required")]
full_rerun_tasks = [e for e in manifest if e.get("rerun_required") and not e.get("eval_only")]
print("eval_only=" + str(len(eval_only_tasks)) + "  full_rerun=" + str(len(full_rerun_tasks)))
assert len(eval_only_tasks) == 1 and len(full_rerun_tasks) == 5, "Manifest counts wrong!"
print("Manifest OK. Starting 6 compute tasks.")

# ── EXECUTE 6 TASKS ───────────────────────────────────────────────────────────
new_results = []

for task in eval_only_tasks + full_rerun_tasks:
    method = task["method"]
    rank   = task["rank"]
    seed   = task["seed"]
    evonly = task.get("eval_only", False)
    tag    = "{}_r{}_seed{}".format(method, rank, seed)

    print("\n" + "#"*60)
    print("# " + ("EVAL ONLY" if evonly else "FULL RERUN") +
          ": {} r={} seed={}".format(method.upper(), rank, seed))
    print("#"*60)

    tm = em = None
    preds = []
    adapter_path = trainable_p = vram = None
    status = "UNKNOWN"
    error_msg = ""

    try:
        set_seed(seed)
        reset_vram()

        expected_adapter = os.path.join(WORKING,
            "{}_r{}_seed{}_steps{}".format(method, rank, seed, MAX_STEPS))

        if evonly:
            if not os.path.isdir(expected_adapter):
                raise FileNotFoundError(
                    "Adapter not found: " + expected_adapter +
                    " (session wiped). Will be treated as FAILED.")
            adapter_path = expected_adapter
            model, processor = load_model_and_processor()
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            eval_data = load_eval_data()
            em, preds = run_evaluation(model, processor, eval_data, tag)
            status = "EVAL_RECOVERED"
            vram = get_peak_vram()
            purge(model)

        else:
            adapter_path = expected_adapter
            os.makedirs(adapter_path, exist_ok=True)
            model, processor = load_model_and_processor()
            tokenizer = get_tokenizer(processor)
            model, trainable_p = inject_adapter(model, method, rank)
            train_data = load_and_prepare_train_data(tokenizer, seed)
            trainer, tm = run_training(model, processor, train_data, seed, adapter_path)
            tm["trainable_params"] = trainable_p
            trainer.model.save_pretrained(adapter_path)
            processor.save_pretrained(adapter_path)
            print("Adapter saved: " + adapter_path)
            shutil.copytree(adapter_path,
                os.path.join(FINAL_DIR, tag + "_adapter"), dirs_exist_ok=True)
            eval_data = load_eval_data()
            em, preds = run_evaluation(model, processor, eval_data, tag)
            status = "NEWLY_RERUN"
            vram = get_peak_vram()
            purge(model)

    except Exception:
        error_msg = traceback.format_exc()
        status = "FAILED"
        print("ERROR:\n" + error_msg)

    new_results.append({
        "method": method, "rank": rank, "seed": seed,
        "accuracy":          em.get("accuracy") if em else None,
        "status":            status,
        "train_loss":        tm.get("train_loss") if tm else None,
        "train_runtime_sec": tm.get("train_runtime_sec") if tm else None,
        "eval_runtime_sec":  em.get("eval_runtime_sec") if em else None,
        "trainable_params":  trainable_p,
        "peak_vram_gb":      vram,
        "artifact_path":     adapter_path if status != "FAILED" else None,
        "error":             error_msg,
    })

    if status != "FAILED":
        save_task(tag, method, rank, seed, tm, em, preds, adapter_path, status, vram)
    save_master(manifest, new_results)
    print_summary(method, rank, seed, em, tm, vram, status)

# ── FINAL OUTPUTS ─────────────────────────────────────────────────────────────
print("Generating final outputs...")
rows = save_master(manifest, new_results)

print("\n--- FINAL 24-RUN TABLE ---")
print("{:<6} {:<4} {:<4} {:>9} {:<22}".format("Method","Rank","Seed","Accuracy","Status"))
print("-"*55)
for row in rows:
    acc = row["accuracy"]
    print("{:<6} {:<4} {:<4} {:>9} {:<22}".format(
        row["method"].upper(), row["rank"], row["seed"],
        "{:.2f}%".format(acc*100) if acc is not None else "N/A",
        str(row["status"])))

shutil.make_archive(os.path.join(WORKING,"final_results"), "zip",
                    WORKING, "final_results")
with zipfile.ZipFile(os.path.join(WORKING,"final_results.zip"), "a") as zf:
    for fn in ["master_results.csv","master_results.json",
               "recovery_status.csv","pilot_recovery_manifest.json"]:
        fp = os.path.join(WORKING, fn)
        if os.path.exists(fp): zf.write(fp, fn)

print("final_results.zip created.")
print("DONE.")
