#!/usr/bin/env python3
"""
Batch inference pipeline for Airbnb NZ reviews (Parquet input -> Parquet output).

- Deception (binary) DistilBERT
- Sentiment (multi-class, e.g. 3-class) DistilBERT
- Chunked Parquet streaming via pyarrow
- CUDA/FP16 when available; deterministic seeds
- Writes all input columns + prediction columns to output
"""

import os
import argparse
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------
# Reproducibility & torch setup
# ----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # we keep deterministic algorithms off for speed
    torch.use_deterministic_algorithms(False)

def get_device_and_dtype(use_fp16_if_cuda=True):
    if torch.cuda.is_available() and not os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        device = torch.device("cuda")
        dtype = torch.float16 if use_fp16_if_cuda else torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype

# ----------------------------
# Model helpers
# ----------------------------
def load_model_and_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tok, mdl

def get_id2label(model) -> Dict[int, str]:
    m = getattr(model.config, "id2label", None)
    if isinstance(m, dict) and m:
        return {int(k): str(v).lower() for k, v in m.items()}
    # sensible fallback for binary or 3-class
    num_labels = getattr(model.config, "num_labels", 2)
    if num_labels == 2:
        return {0: "negative", 1: "positive"}
    if num_labels == 3:
        return {0: "negative", 1: "netural", 2: "positive"}
    return {i: f"class_{i}" for i in range(num_labels)}

def find_positive_index(id2label: Dict[int, str]) -> int:
    """Pick the 'positive/deceptive' index for binary models."""
    hints = {"deceptive", "positive", "spam", "fake", "pos"}
    for i, name in id2label.items():
        if name in hints:
            return i
    # fallback: class 1
    return 1

@torch.no_grad()
def predict_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    dtype: torch.dtype,
    max_len: int = 256,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (logits, probs). For binary models, probs has shape [N, 2].
    For multi-class, probs is softmax over classes.
    """
    model = model.to(device)
    model.eval()

    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if dtype == torch.float16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)
        logits = out.logits.detach().to("cpu")
        all_logits.append(logits)
    logits = torch.cat(all_logits, dim=0).numpy()

    # convert to probabilities
    if logits.shape[1] == 1:
        # sigmoid -> probs for class1; create 2-col probs
        p1 = 1.0 / (1.0 + np.exp(-logits))
        probs = np.hstack([1.0 - p1, p1])
    else:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)

    return logits, probs

# --- Compatible Parquet streaming function (no .scan() dependency) ---
def parquet_batches(parquet_path: str, columns: List[str] | None, batch_rows: int = 50_000):
    """
    Stream a Parquet file in row batches as pandas DataFrames.
    If columns is None, read all columns.
    """
    pf = pq.ParquetFile(parquet_path)
    for rb in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield rb.to_pandas()

# ----------------------------
# Main inference routine
# ----------------------------
def run_inference(
    input_parquet: str,
    output_path: str,
    deception_model_dir: str,
    sentiment_model_dir: str,
    text_col: str = "text",
    id_cols: List[str] | Tuple[str, ...] = ("review_id", "listing_id"),
    max_len: int = 256,
    batch_size: int = 256,
    deception_threshold: float = 0.81,  
    use_fp16_if_cuda: bool = True,
    batch_rows: int = 50_000,
    partition_by: List[str] | None = None,  # e.g., ["year", "month"]
    read_columns: List[str] | None = None,  # None -> read ALL columns
):
    """
    output_path:
        - If partition_by is None: a single .parquet file path.
        - If partition_by is not None: a directory to write a partitioned dataset.
    """
    assert os.path.exists(input_parquet), f"Missing input: {input_parquet}"
    out_dir = output_path if partition_by else os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    set_seed(42)
    device, dtype = get_device_and_dtype(use_fp16_if_cuda=use_fp16_if_cuda)

    # Load models
    dec_tok, dec_mdl = load_model_and_tokenizer(deception_model_dir)
    sen_tok, sen_mdl = load_model_and_tokenizer(sentiment_model_dir)

    dec_id2label = get_id2label(dec_mdl)
    sen_id2label = get_id2label(sen_mdl)
    dec_pos_idx = find_positive_index(dec_id2label)

    # Writer & counters
    writer = None
    total_rows = 0
    t0 = time.time()

    for df in parquet_batches(input_parquet, columns=read_columns, batch_rows=batch_rows):
        # --- required column check ---
        if text_col not in df.columns:
            raise ValueError(f"Missing required text column '{text_col}' in input parquet.")

        # --- texts for tokenization (do not mutate df types) ---
        texts = df[text_col].astype("string").fillna("").astype(str).tolist()

        # --- Deception (binary or 2-class) ---
        d_logits, d_probs = predict_texts(
            texts, dec_tok, dec_mdl, device, dtype, max_len=max_len, batch_size=batch_size
        )
        d_logits = d_logits.astype(np.float32)
        d_probs = d_probs.astype(np.float32)

        # probability of the 'positive/deceptive' class
        d_p_pos = d_probs[:, dec_pos_idx]
        d_label = (d_p_pos >= deception_threshold).astype(np.int64)

        # name deception prob columns by id2label order
        dec_prob_cols = {f"deception_prob_{dec_id2label[i]}": d_probs[:, i] for i in range(d_probs.shape[1])}

        # --- Sentiment (multi-class) ---
        s_logits, s_probs = predict_texts(
            texts, sen_tok, sen_mdl, device, dtype, max_len=max_len, batch_size=batch_size
        )
        s_logits = s_logits.astype(np.float32)
        s_probs = s_probs.astype(np.float32)
        s_label = np.argmax(s_probs, axis=1).astype(np.int64)

        sent_prob_cols = {f"sent_prob_{sen_id2label[i]}": s_probs[:, i] for i in range(s_probs.shape[1])}

        # --- Assemble result chunk: keep ALL input columns ---
        out = df.copy()  # preserves original dtypes
        # logits (name by index to avoid dependency on label names)
        for j in range(d_logits.shape[1]):
            out[f"deception_logit_{j}"] = d_logits[:, j]
        for j in range(s_logits.shape[1]):
            out[f"sent_logit_{j}"] = s_logits[:, j]
        # probs + preds
        for k, v in dec_prob_cols.items():
            out[k] = v
        for k, v in sent_prob_cols.items():
            out[k] = v
        out["deception_pred"] = d_label
        out["sent_pred"] = s_label

        # --- Write out (single file or partitioned dataset) ---
        table = pa.Table.from_pandas(out, preserve_index=False)

        if partition_by:
            # write as a partitioned dataset to a directory
            import pyarrow.dataset as pads
            pads.write_dataset(
                table,
                base_dir=output_path,
                format="parquet",
                partitioning=partition_by,
                existing_data_behavior="overwrite_or_ignore"
            )
        else:
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)

        total_rows += len(out)

    if writer is not None:
        writer.close()

    dt = time.time() - t0
    rps = total_rows / dt if dt > 0 else float("inf")
    print(f"[DONE] Wrote {total_rows:,} rows to {output_path} in {dt:.1f}s ({rps:,.0f} rows/s).")

    # --- Validation pass ---
    try:
        if partition_by:
            import pyarrow.dataset as pads
            ds = pads.dataset(output_path, format="parquet")
            cols = {f.name for f in ds.schema}
            # sample row count
            n = sum(1 for _ in ds.scan(columns=[]).to_batches())  # avoids loading all
            # ensure key prediction columns exist
            needed = {"deception_pred", "sent_pred"}
            assert needed.issubset(cols), f"Missing prediction columns: {needed - cols}"
        else:
            res = pq.read_table(output_path, columns=["deception_pred", "sent_pred"])
            n = res.num_rows
            assert res.column("deception_pred").null_count == 0
            assert res.column("sent_pred").null_count == 0
        print("[VALIDATION] Output OK; predictions present and non-null.")
    except Exception as e:
        print(f"[VALIDATION WARNING] Skipped or partial validation due to: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True, help="Cleaned Airbnb reviews (Parquet).")
    ap.add_argument("--output_path", required=True,
                    help="Single file (.parquet) or output directory (when --partition_by is used).")
    ap.add_argument("--deception_model_dir", required=True)
    ap.add_argument("--sentiment_model_dir", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--batch_rows", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--deception_threshold", type=float, default=0.81)
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU and float32.")
    ap.add_argument("--partition_by", nargs="*", default=None,
                    help='Partition columns, e.g. --partition_by year month. Output is a directory.')
    ap.add_argument("--read_columns", nargs="*", default=None,
                    help='Columns to read from input. Omit to read ALL columns.')
    args = ap.parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

    run_inference(
        input_parquet=args.input_parquet,
        output_path=args.output_path,
        deception_model_dir=args.deception_model_dir,
        sentiment_model_dir=args.sentiment_model_dir,
        text_col=args.text_col,
        max_len=args.max_len,
        batch_size=args.batch_size,
        deception_threshold=args.deception_threshold,
        use_fp16_if_cuda=not args.cpu_only,
        batch_rows=args.batch_rows,
        partition_by=args.partition_by,
        read_columns=args.read_columns,  # None -> all input columns
    )

if __name__ == "__main__":
    main()
