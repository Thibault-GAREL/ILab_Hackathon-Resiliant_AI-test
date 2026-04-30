#!/usr/bin/env python3
"""
Test de précision + énergie pour Gemma 4 E2B (GGUF)
pip install llama-cpp-python codecarbon rouge-score datasets tqdm
"""

import json, time, argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset
from codecarbon import EmissionsTracker
from rouge_score import rouge_scorer as rouge_lib

from huggingface_hub import hf_hub_download

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_MODELS = {
    "bf16":    "google/gemma-4-E2B-it",                    # via transformers
    "q8":      "ggml-org/gemma-4-E2B-it-GGUF",
    "q4_k_m":  "bartowski/google_gemma-4-E2B-it-GGUF",
}

# ─── Chargement modèle ────────────────────────────────────────────────────────

def load_gguf_model(gguf_path: str, n_gpu_layers: int = -1):
    from llama_cpp import Llama
    print(f"  Chargement GGUF : {gguf_path}")
    return Llama(
        model_path=gguf_path,
        n_gpu_layers=n_gpu_layers,   # -1 = tout sur GPU, 0 = CPU only
        n_ctx=8192,
        n_batch=512,
        verbose=False,
    )

def load_hf_model(model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print(f"  Chargement HF : {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tok, mdl

# ─── Inférence unifiée ────────────────────────────────────────────────────────

def infer_gguf(model, prompt: str, max_tokens: int = 256) -> tuple[str, float]:
    """Retourne (texte_généré, latence_ms)"""
    # Filet de sécurité : tronque le prompt si prompt + max_tokens dépasse n_ctx
    n_ctx = model.n_ctx()
    max_prompt_tokens = n_ctx - max_tokens - 16  # marge pour tokens spéciaux
    tokens = model.tokenize(prompt.encode("utf-8"))
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[:max_prompt_tokens]
        prompt = model.detokenize(tokens).decode("utf-8", errors="ignore")

    t0 = time.perf_counter()
    out = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,       # déterministe pour reproductibilité
        echo=False,
    )
    latency = (time.perf_counter() - t0) * 1000
    return out["choices"][0]["text"].strip(), latency

def infer_hf(tokenizer, model, prompt: str, max_tokens: int = 256) -> tuple[str, float]:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                              do_sample=False)
    latency = (time.perf_counter() - t0) * 1000
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return text.strip(), latency

# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(prediction: str, reference: str,
                    task_type: str) -> dict:
    metrics = {}

    if task_type in ("generation", "summarization", "qa"):
        scorer = rouge_lib.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, prediction)
        metrics["rouge1"]  = round(scores["rouge1"].fmeasure, 4)
        metrics["rouge2"]  = round(scores["rouge2"].fmeasure, 4)
        metrics["rougeL"]  = round(scores["rougeL"].fmeasure, 4)

    if task_type in ("classification", "qa_exact"):
        # Exact match (insensible à la casse)
        pred_clean = prediction.strip().lower()
        ref_clean  = reference.strip().lower()
        metrics["exact_match"] = int(pred_clean == ref_clean)
        # Soft match : la réponse contient la référence
        metrics["contains"]    = int(ref_clean in pred_clean)

    if task_type == "perplexity":
        # Placeholder — calculé séparément (voir compute_perplexity)
        metrics["perplexity"] = None

    return metrics

def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Calcule la perplexité moyenne sur une liste de textes (modèle HF uniquement)"""
    import torch, math
    total_nll, total_tokens = 0.0, 0
    model.eval()
    for text in tqdm(texts, desc="  Perplexité"):
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        n = enc["input_ids"].shape[1]
        total_nll    += loss.item() * n
        total_tokens += n
    return round(math.exp(total_nll / total_tokens), 3)

# ─── Chargement dataset ───────────────────────────────────────────────────────

def load_eval_dataset(dataset_config: dict, n_samples: int) -> list[dict]:
    """
    dataset_config exemples :
      {"source": "cnn_dailymail", "version": "3.0.0", "split": "test",
       "prompt_col": "article", "label_col": "highlights", "task": "summarization"}

      {"source": "squad", "split": "validation",
       "prompt_col": "question", "context_col": "context",
       "label_col": "answers.text.0", "task": "qa_exact"}

      {"source": "json", "path": "mon_dataset.jsonl",
       "prompt_col": "input", "label_col": "output", "task": "generation"}
    """
    src = dataset_config["source"]

    if src == "json":
        ds = Dataset.from_json(dataset_config["path"])
    elif src == "csv":
        ds = Dataset.from_csv(dataset_config["path"])
    else:
        kwargs = {"split": dataset_config.get("split", "test")}
        if "version" in dataset_config:
            ds = load_dataset(src, dataset_config["version"], **kwargs)
        else:
            ds = load_dataset(src, **kwargs)

    ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for row in ds:
        prompt_col = dataset_config["prompt_col"]
        label_col  = dataset_config["label_col"]

        prompt = str(row[prompt_col])

        # Support chemin imbriqué ex: "answers.text.0"
        ref = row
        for part in label_col.split("."):
            if isinstance(ref, dict):
                ref = ref.get(part, "")
            elif isinstance(ref, list) and part.isdigit():
                ref = ref[int(part)] if int(part) < len(ref) else ""
            else:
                ref = str(ref)

        # Ajout contexte si Q&A
        if "context_col" in dataset_config:
            context = str(row[dataset_config["context_col"]])
            prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"

        samples.append({
            "prompt": prompt,
            "reference": str(ref),
            "task": dataset_config.get("task", "generation"),
        })

    return samples

# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_evaluation(
    model_path: str,
    dataset_config: dict,
    n_samples: int    = 100,
    max_tokens: int   = 256,
    use_gguf: bool    = True,
    n_gpu_layers: int = -1,
    output_file: str  = None,
):
    print(f"\n{'═'*60}")
    print(f"  ÉVALUATION GEMMA 4 E2B")
    print(f"  Modèle  : {model_path}")
    print(f"  Dataset : {dataset_config['source']} ({n_samples} samples)")
    print(f"{'═'*60}\n")

    # Chargement modèle
    if use_gguf:
        model = load_gguf_model(model_path, n_gpu_layers)
        tokenizer = None
    else:
        tokenizer, model = load_hf_model(model_path)

    # Chargement dataset
    print("  Chargement dataset...")
    samples = load_eval_dataset(dataset_config, n_samples)
    print(f"  {len(samples)} exemples chargés\n")

    results    = []
    all_energy = []
    all_co2    = []

    for i, sample in enumerate(tqdm(samples, desc="  Évaluation")):
        prompt    = sample["prompt"]
        reference = sample["reference"]
        task      = sample["task"]

        # ── Mesure énergie par réponse ──
        tracker = EmissionsTracker(
            project_name="gemma4_eval",
            log_level="error",
            save_to_file=False,
            tracking_mode="process",
        )
        tracker.start()

        if use_gguf:
            prediction, latency_ms = infer_gguf(model, prompt, max_tokens)
        else:
            prediction, latency_ms = infer_hf(tokenizer, model, prompt, max_tokens)

        emissions_kg = tracker.stop()  # kg CO₂ eq.
        energy_kwh   = tracker._total_energy.kWh
        energy_wh    = energy_kwh * 1000
        co2_g        = emissions_kg * 1000

        # ── Métriques de précision ──
        precision_metrics = compute_metrics(prediction, reference, task)

        result = {
            "index":       i,
            "task":        task,
            "prediction":  prediction[:200],  # tronqué pour lisibilité
            "reference":   reference[:200],
            "latency_ms":  round(latency_ms, 1),
            "energy_wh":   round(energy_wh, 6),
            "co2_g":       round(co2_g, 6),
            **precision_metrics,
        }
        results.append(result)
        all_energy.append(energy_wh)
        all_co2.append(co2_g)

    # ── Résumé ──
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "modele":              model_path,
        "dataset":             dataset_config["source"],
        "n_samples":           len(results),
        "─── PRÉCISION ───":   {},
        "rouge1":              avg("rouge1"),
        "rouge2":              avg("rouge2"),
        "rougeL":              avg("rougeL"),
        "exact_match":         avg("exact_match"),
        "contains":            avg("contains"),
        "─── PERFORMANCE ─":  {},
        "latence_moy_ms":      avg("latency_ms"),
        "latence_p95_ms":      round(sorted(r["latency_ms"] for r in results)[int(len(results)*0.95)], 1),
        "─── ÉNERGIE ────":   {},
        "energie_moy_wh":      round(sum(all_energy) / len(all_energy), 6),
        "energie_total_wh":    round(sum(all_energy), 4),
        "co2_moy_g":           round(sum(all_co2) / len(all_co2), 6),
        "co2_total_g":         round(sum(all_co2), 4),
        "wh_par_1000_req":     round(sum(all_energy) / len(all_energy) * 1000, 2),
    }

    print(f"\n{'═'*60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{'═'*60}\n")

    if output_file:
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "details": results},
                      f, indent=2, ensure_ascii=False)
        print(f"  Résultats sauvegardés → {output_file}")

    return summary, results


# ─── Comparaison multi-modèles ────────────────────────────────────────────────

def compare_models(models: dict, dataset_config: dict, n_samples: int = 50):
    """
    Évalue plusieurs variantes et affiche un tableau comparatif.
    models = {"Q4_K_M": "/path/q4.gguf", "Q8_0": "/path/q8.gguf", ...}
    """
    all_summaries = {}

    for name, path in models.items():
        use_gguf = path.endswith(".gguf")
        summary, _ = run_evaluation(
            model_path=path,
            dataset_config=dataset_config,
            n_samples=n_samples,
            use_gguf=use_gguf,
            output_file=f"results_{name}.json",
        )
        all_summaries[name] = summary

    # Tableau comparatif
    print(f"\n{'═'*80}")
    print(f"  TABLEAU COMPARATIF")
    print(f"{'═'*80}")
    header = f"{'Modèle':<12} {'ROUGE-L':>8} {'Exact':>7} {'Lat(ms)':>9} {'Énergie/req(Wh)':>16} {'CO2/req(g)':>11}"
    print(header)
    print("─" * 80)

    ref_rouge = None
    for name, s in all_summaries.items():
        rouge  = s.get("rougeL") or 0
        exact  = s.get("exact_match") or 0
        lat    = s.get("latence_moy_ms") or 0
        energy = s.get("energie_moy_wh") or 0
        co2    = s.get("co2_moy_g") or 0

        if ref_rouge is None:
            ref_rouge = rouge
            delta = ""
        else:
            delta = f"  ({(rouge/ref_rouge-1)*100:+.1f}%)" if ref_rouge else ""

        print(f"  {name:<12} {rouge:>8.4f}{delta:<12} {exact:>7.4f} "
              f"{lat:>9.1f} {energy:>16.6f} {co2:>11.6f}")

    return all_summaries


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Exemple d'utilisation directe ──
    # Adapte ces paramètres à ton cas

    # Dataset : remplace par le tien
    # task options: "generation", "summarization", "qa_exact", "classification"
    DATASET_CONFIG = {
        "source":     "cnn_dailymail",  # ou "squad", "json", "csv"
        "version":    "3.0.0",          # pour cnn_dailymail
        "split":      "test",
        "prompt_col": "article",
        "label_col":  "highlights",
        "task":       "summarization",
    }

    model_path = hf_hub_download(
        repo_id="unsloth/gemma-4-E2B-it-GGUF",
        filename="gemma-4-E2B-it-Q4_K_M.gguf",  # nom exact du fichier
    )

    # ── Option 1 : tester UN modèle GGUF ──
    run_evaluation(
        model_path   = model_path,
        dataset_config = DATASET_CONFIG,
        n_samples    = 100,
        max_tokens   = 128,
        use_gguf     = True,
        output_file  = "results_q4km.json",
    )

    # ── Option 2 : comparer plusieurs quantifications ──
    # compare_models(
    #     models={
    #         "Q4_K_M": "./gemma4-e2b/google_gemma-4-E2B-it-Q4_K_M.gguf",
    #         "Q5_K_M": "./gemma4-e2b/google_gemma-4-E2B-it-Q5_K_M.gguf",
    #         "Q8_0":   "./gemma4-e2b/google_gemma-4-E2B-it-Q8_0.gguf",
    #     },
    #     dataset_config=DATASET_CONFIG,
    #     n_samples=50,
    # )
