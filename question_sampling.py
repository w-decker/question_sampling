"""Embedding, representational similarity, and farthest point sampling helpers."""
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MODELS = {
    "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "gpt2": "openai-community/gpt2",
    "llama": "meta-llama/Llama-3.2-3B",
}


def read_questions(path="questions.txt"):
    """Keep nonempty, unique questions in file order and their 1-based line numbers."""
    unique = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if question := line.strip():
            unique.setdefault(question, line_number)
    if len(unique) < 3:
        raise ValueError("At least three unique questions are required for RSA.")
    return np.array(list(unique)), np.array(list(unique.values()))


def normalize(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise ValueError("Embeddings must be finite and nonzero.")
    return embeddings / norms


def embed_questions(questions, model_id, batch_size=8, device=None):
    """Encode sentences, or mean-pool final token states excluding padding/specials."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if model_id.startswith("sentence-transformers/"):
        model = SentenceTransformer(model_id, device=device)
        lengths = [len(model.tokenizer.encode(q)) for q in questions]
        if max(lengths) > model.max_seq_length:
            raise ValueError("A question exceeds the Sentence Transformer context length.")
        return normalize(model.encode(list(questions), batch_size=batch_size,
                                      show_progress_bar=True, convert_to_numpy=True))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device).eval()
    context = getattr(model.config, "max_position_embeddings", None)
    pooled = []
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            inputs = tokenizer(list(questions[start:start + batch_size]), padding=True,
                               return_tensors="pt", return_special_tokens_mask=True)
            special = inputs.pop("special_tokens_mask").to(device)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if context is not None and inputs["input_ids"].shape[1] > context:
                raise ValueError("A question exceeds the model context length.")
            hidden = model(**inputs, use_cache=False).last_hidden_state.float()
            mask = (inputs["attention_mask"] * (1 - special)).unsqueeze(-1)
            counts = mask.sum(dim=1)
            if (counts == 0).any():
                raise ValueError("A question has no content tokens.")
            pooled.append(((hidden * mask).sum(dim=1) / counts).cpu().numpy())
    return normalize(np.concatenate(pooled))


def save_embeddings(name, questions, line_numbers, output_dir="outputs", **kwargs):
    """Save one self-contained NPZ per model; reuse only matching cached inputs."""
    model_id = MODELS[name]
    path = Path(output_dir) / f"{name}.npz"
    if path.exists():
        with np.load(path) as data:
            if (str(data["model_id"]) != model_id
                    or not np.array_equal(data["questions"], questions)
                    or not np.array_equal(data["line_numbers"], line_numbers)):
                raise ValueError(f"{path} differs from current inputs; move it before rerunning.")
            return data["embeddings"].copy()
    embeddings = embed_questions(questions, model_id, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, questions=questions,
                        line_numbers=line_numbers, model_id=model_id,
                        pooling="sentence-transformer native; otherwise final-layer content-token mean",
                        normalization="L2")
    return embeddings


def cosine_similarity(embeddings):
    unit = normalize(embeddings)
    return np.clip(unit @ unit.T, -1, 1)


def representational_similarity(similarities):
    """Spearman rho on unique off-diagonal question pairs; no independence p-values."""
    names = list(similarities)
    matrices = [similarities[name] for name in names]
    shape = matrices[0].shape
    if shape[0] < 3 or any(m.shape != shape or m.ndim != 2 or m.shape[0] != m.shape[1]
                           for m in matrices):
        raise ValueError("Similarity matrices must be aligned square matrices with >=3 rows.")
    triangle = np.triu_indices(shape[0], k=1)
    vectors = [matrix[triangle] for matrix in matrices]
    if any(not np.isfinite(v).all() or np.ptp(v) == 0 for v in vectors):
        raise ValueError("RSA requires finite, nonconstant pairwise similarities.")
    return np.array([[spearmanr(a, b).statistic for b in vectors] for a in vectors])


def farthest_point_sampling(embeddings, n_samples, seed=0):
    """PyVista example's greedy Euclidean FPS, with unique-index tie handling."""
    points = normalize(embeddings)
    if not isinstance(n_samples, (int, np.integer)) or not 1 <= n_samples <= len(points):
        raise ValueError("n_samples must be an integer between 1 and the question count.")
    sampled = np.empty(n_samples, dtype=int)
    sampled[0] = np.random.default_rng(seed).integers(len(points))
    distances = np.full(len(points), np.inf)
    for i in range(1, n_samples):
        last = points[sampled[i - 1]]
        distances = np.minimum(distances, np.linalg.norm(points - last, axis=1))
        distances[sampled[:i]] = -np.inf
        sampled[i] = np.argmax(distances)
    return sampled
