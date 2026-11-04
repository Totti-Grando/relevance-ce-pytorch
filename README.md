# relevance-ce-pytorch

Weights for the relevance cross-encoder: 9-way classification of (question, claim) pairs into
IRRELEVANT, UNRESOLVED, ECOSYSTEM, WRONG_RELATION, WRONG_INTENT, WRONG_TIME, DIRECT, PARTIAL, CONTEXT.
Backbone is `cross-encoder/ettin-reranker-150m-v1` (ModernBERT, 150M) with a Dense/LayerNorm/Linear head.

## Checkpoints

| dir | release | status | test acc | macro-F1 | missed | fooled |
|---|---|---|---|---|---|---|
| `model-150m-mh-cal` | [v2](https://github.com/Totti-Grando/relevance-ce-pytorch/releases/tag/v2) | **current best** | 0.919 | 0.877 | 7.9% | 2.1% |
| `model-150m-all` | [v1](https://github.com/Totti-Grando/relevance-ce-pytorch/releases/tag/v1) | previous | 0.863 | 0.843 | 14.2% | 3.0% |

"missed" = DIRECT/PARTIAL claims labeled non-relevant; "fooled" = non-relevant claims labeled relevant.
The two rows are on different test splits (v3 data for mh-cal, v2 data for model-150m-all re-scored),
so treat the gap as indicative. `model-150m-mh-cal` was trained on synthetic financial pairs plus
HotpotQA / MuSiQue / 2WikiMultiHop converted with an entity-link rule (claims that reach the question
only through an unnamed bridge entity are UNRESOLVED), has a head calibrated by a 9x9 affine edit of the
output layer, and resolves same-metric / different-firm claims as ECOSYSTEM.

## Layout
```
model-150m-mh-cal/
  encoder/            # HF ModernBERT backbone (config + tokenizer; model.safetensors from the release)
  head.pt             # dense/norm/out head + temperature (git LFS, 2.3 MB)
  meta.json           # labels, utility map, temperature, max_length, head_edit, quantization
  report.json         # val/test scorecards
model-150m-all/       # previous checkpoint, same layout (v1)
common.py             # model definition + load_model/format_pair (copied from the training repo)
```

## Get the weights

`model.safetensors` (596 MB) is not in git at all; `head.pt` is in git LFS. Download the release
assets (plain HTTPS, works with curl or Python):

```bash
git clone https://github.com/Totti-Grando/relevance-ce-pytorch.git
cd relevance-ce-pytorch
base=https://github.com/Totti-Grando/relevance-ce-pytorch/releases/download/v2
curl -L -o model-150m-mh-cal/encoder/model.safetensors $base/model.safetensors
curl -L -o model-150m-mh-cal/head.pt $base/head.pt          # only needed without git-lfs
```
Or in pure Python (no curl needed):
```python
import urllib.request
base = "https://github.com/Totti-Grando/relevance-ce-pytorch/releases/download/v2"
urllib.request.urlretrieve(f"{base}/model.safetensors", "model-150m-mh-cal/encoder/model.safetensors")
urllib.request.urlretrieve(f"{base}/head.pt", "model-150m-mh-cal/head.pt")
```

The v2 release also carries ONNX files for CPU serving without PyTorch:

| asset | what | notes |
|---|---|---|
| `model.onnx` | fp32 graph, opset 17 | 599 MB; fastest accurate option on CPU |
| `model.int8.onnx` | weight-only 8-bit (MatMulNBits, block 128, fp32 activations) | 270 MB; 99.7% label agreement with fp32 |
| `tokenizer.zip` | HF tokenizer files for the ONNX inputs | same tokenizer as `encoder/` |

Do not quantize this model with `onnxruntime.quantization.quantize_dynamic`: per-tensor int8
activations cost about 8 points of accuracy and wipe out the ECOSYSTEM class. Inputs are
`input_ids`/`attention_mask` (int64, `[Q] question tags` paired with `[C] claim tags`, max length 128);
output is `logits` `[batch, 9]`, divide by `meta.json["temperature"]` before softmax.

For the v1 assets (`model-150m-all`), replace `v2` with `v1` and the directory name accordingly.

## Install (Intel GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu   # XPU-enabled build
pip install "transformers>=4.51" safetensors
```

## Run
```python
import torch
from common import load_model, format_pair

print("XPU:", torch.xpu.is_available())          # True on the Intel GPU
m, tok, meta = load_model("model-150m-mh-cal")
m = m.to("xpu").eval()

rows = [{"question": "What were Northwind's net flows in Q2 2026?",
         "claim": "Northwind took in $1.2B in Q2 2026.", "q_tags": "", "c_tags": ""}]
q, c = zip(*[format_pair(r) for r in rows])
enc = tok(list(q), list(c), padding=True, truncation="longest_first",
          max_length=meta["max_length"], return_tensors="pt").to("xpu")
with torch.no_grad():
    probs = torch.softmax(m(enc["input_ids"], enc["attention_mask"]) / meta["temperature"], -1)
print([meta["labels"][i] for i in probs.argmax(-1)])   # -> ['DIRECT']
```

Labels and their relevance utilities live in `meta.json`; `DIRECT`/`PARTIAL` are the relevant
classes. If `torch.xpu.is_available()` is `False`, update the Intel GPU driver/runtime (a driver
issue, not a pip package). CPU/CUDA work too — just change `"xpu"` to `"cpu"`/`"cuda"`.
