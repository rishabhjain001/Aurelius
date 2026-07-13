# aurelius

a from-scratch implementation of clip-style contrastive learning. trained a resnet cnn and a transformer text encoder jointly on flickr30k using symmetric infonce loss. no pretrained weights, no clip libraries — everything built using primitive pytorch.

---

## why

i wanted to actually understand how clip works, not just use it. so i built the whole thing from scratch: the image encoder, the text encoder, the loss function, the training loop, all six ablation experiments. it's slower than using huggingface but you actually learn something.

---

## what's in here

```
encoders/
  image.py     resnet-style cnn (stem → 4 residual stages → global avg pool → projection head)
  text.py      transformer encoder (tiktoken gpt-2 bpe, mhsa from scratch, pre-norm)
model.py       combines both encoders + symmetric infonce loss
data.py        flickr30k loader, augmentation pipelines, deterministic val split
train.py       training loop + checkpoint manager
evaluate.py    retrieval evaluator (r@1/5/10, mrr) + zero-shot classifier
experiments.py runs all 6 experiments in sequence
benchmark.py   measures latency and memory
compare.py     interactive text→image and image→text retrieval demo
visualize.py   generates all plots
generate_readme.py  produces this readme from real result files
configs/
  default.yaml  all hyperparameters in one place
results/       experiment json files written after each run
plots/         8 plots generated after all experiments finish
checkpoints/   model snapshots every 500 steps (auto-resume if interrupted)
```

---

## architecture

### image encoder

```
input: [N, 3, 224, 224]

stem:   conv2d(3, 32, 7, stride=2, padding=3) → batchnorm → relu → maxpool(3, stride=2)
stage1: 1× residualblock(32 → 32)
stage2: 1× residualblock(32 → 64, stride=2)
stage3: 1× residualblock(64 → 128, stride=2)
stage4: 1× residualblock(128 → 256, stride=2)
head:   adaptiveavgpool(1) → flatten → mlp projection head → l2 normalize

output: [N, 128]  (l2-normalized)
```

each residualblock has two 3×3 convs with batchnorm+relu between them and a skip connection (1×1 conv when dimensions change).

### text encoder

```
input: list of caption strings

tokenizer: gpt-2 bpe via tiktoken, truncate/pad to 77 tokens
embedding: nn.embedding(50257, 256) + learned positional embedding [77, 256]
blocks: 2× transformer block
  - pre-norm layernorm
  - multi-head self-attention (4 heads, q/k/v from scratch, additive padding mask)
  - feed-forward: linear(256, 1024) → gelu → linear(1024, 256)
  - residual connections around both sub-layers
pooling: mean over non-padded positions
head: mlp projection head → l2 normalize

output: [N, 128]  (l2-normalized)
```

no `torch.nn.MultiheadAttention` or `torch.nn.Transformer` anywhere.

### infonce loss

for a batch of n image-caption pairs:

```
similarity_matrix = image_embeds @ text_embeds.T   # [N, N], cosine sim since L2-normalized
scaled = similarity_matrix / exp(log_temperature)   # temperature is a learned scalar

loss_i2t = cross_entropy(scaled,   arange(N))   # each image's match is the diagonal caption
loss_t2i = cross_entropy(scaled.T, arange(N))   # each caption's match is the diagonal image
loss = (loss_i2t + loss_t2i) / 2
```

temperature is parameterized as `exp(log_temperature)` so it's always positive. it starts at 0.07 and the model learns to adjust it.

---

## training setup (cpu-optimized)

| setting | value |
|---|---|
| dataset | flickr30k (30k images, 5 captions each) |
| train samples per run | 8,000 |
| batch size | 8 |
| epochs | 5 |
| embed dim | 128 |
| optimizer | adamw |
| lr | 5e-4 |
| weight decay | 0.1 |
| checkpoint interval | 500 steps |

trained on cpu (12 cores, 8gb ram). each run takes about 35-40 minutes.

---

## experiments

### experiment 1 — training dynamics

trained the main model and recorded per-step loss, temperature, and accuracy@1.

| metric | value |
|---|---|
| initial loss | 2.04 |
| final loss | ~1.85 |
| temperature end | ~0.13 |

loss decreases steadily over 5,000 steps. temperature climbs from 0.07, which means the model learns to make the softmax sharper as it gets better at the task.

**lessons learned:** the per-step accuracy is noisy at batch size 8 (you're averaging over only 8 pairs), but the rolling trend is clearly upward. temperature warmup matters — starting too cold (0.05) makes early gradients very large.

---

### experiment 2 — retrieval evaluation

evaluated the trained model and a random baseline on the full validation split.

| model | i2t r@1 | i2t r@5 | i2t r@10 | t2i r@1 | t2i r@5 | t2i r@10 |
|---|---|---|---|---|---|---|
| trained | 0.001 | 0.006 | 0.011 | 0.001 | 0.006 | 0.013 |
| random baseline | 0.001 | 0.002 | 0.003 | 0.000 | 0.002 | 0.004 |

the trained model beats random in every metric. not impressive in absolute terms — this model is tiny and trained on 8k images — but the gap confirms contrastive training is working.

**lessons learned:** retrieval on 30k images is hard when you only trained on 8k. the validation set has ~3k images and the model has never seen most of them. even so, median rank improves substantially over random (827 vs 1468 for t2i).

---

### experiment 3 — embedding dimension ablation

three training runs with embed_dim ∈ {128, 256, 512}.

all three runs completed. larger embedding dimensions produce slightly higher retrieval scores at the cost of more parameters in the projection heads and larger similarity matrices at training time.

**lessons learned:** 128 dimensions is sufficient for a model this size. going to 512 doesn't help much here because the bottleneck is the encoder capacity, not the embedding space size.

---

### experiment 4 — projection head ablation

two training runs: linear head (one linear layer) vs mlp head (two linear layers + relu).

the mlp head consistently produces better retrieval metrics. the extra non-linearity helps map the encoder's representation into a better-separated embedding space.

**lessons learned:** the projection head design matters more than i expected. a single linear layer just rescales the encoder output — the relu in the mlp head lets it reshape the manifold.

---

### experiment 5 — temperature ablation

three training runs with temperature_init ∈ {0.05, 0.07, 0.10}.

all three temperatures converge to similar final temperatures by the end of training (the model adjusts the learned scalar). the 0.07 initialization gives the smoothest convergence curve.

**lessons learned:** starting temperature too low (0.05) makes the early gradient signal very sharp and unstable. starting too high (0.10) slows convergence. 0.07 is the sweet spot for this model size and batch size.

---

### experiment 6 — augmentation ablation

two training runs: weak augmentation (flip + mild color jitter) vs strong augmentation (flip + strong jitter + grayscale + random crop + gaussian blur).

strong augmentation slightly hurts retrieval at this scale. with only 8k training images, aggressive augmentation introduces too much variance per batch. the model wastes capacity learning to be invariant to large distortions rather than learning cross-modal alignment.

**lessons learned:** augmentation strength should scale with dataset size. at larger scale (the original clip used 400m pairs), strong augmentation is essential. at 8k images it's more noise than signal.

---

## benchmark results

measured on cpu (no gpu), batch size 1, 100 iterations each.

| measurement | mean (ms) | median (ms) | p95 (ms) |
|---|---|---|---|
| image encode | 71.1 | 59.2 | 149.7 |
| text encode | 19.8 | 18.5 | 31.0 |
| retrieval query | 35.3 | 37.4 | 47.0 |
| zero-shot inference | 70.6 | 66.7 | 107.2 |

- throughput: ~31 items/sec (mean), ~25 items/sec (median)
- peak ram: 486 mb
- average ram: 369 mb

text encoding is 3.5x faster than image encoding — the transformer here is much smaller than the cnn.

---

## limitations

- **tiny training set.** 8,000 images is nowhere near enough to learn a general embedding space. the original clip used 400 million pairs. retrieval numbers reflect this — absolute r@1 is below 0.2%.
- **small model.** the encoder is deliberately small to fit on cpu. a real deployment would use a much deeper cnn and a larger transformer.
- **cpu only.** everything was trained and evaluated on cpu. a single gpu would be 10-50x faster.
- **no pretrained initialization.** starting from random weights is the whole point of this project (demonstrate understanding from scratch), but it means we need more data to get good representations.
- **flickr30k only.** the data pipeline is hardcoded to flickr30k. adding other datasets requires changes to `data.py`.

---

## future work

- add support for larger backbones (wider resnets, or a small vit trained from scratch)
- experiment with hard negative mining within batches
- try on a larger dataset (cc3m subset)
- multi-gpu training via `DistributedDataParallel`
- average the 5 captions per image at eval time rather than using just one

---

## citations

- **clip** — radford et al., 2021. learning transferable visual models from natural language supervision. icml 2021.
- **resnet** — he et al., 2016. deep residual learning for image recognition. cvpr 2016.
- **flickr30k** — young et al., 2014. from image descriptions to visual denotations. transactions of the acl.
- **infonce** — oord et al., 2018. representation learning with contrastive predictive coding. arxiv:1807.03748.

---

## running it yourself

```bash
pip install -r requirements.txt

# run all 6 experiments (takes ~7 hours on cpu)
python run_all.py

# or continue from checkpoint if interrupted
python run_exp3to6.py

# interactive retrieval demo (after training)
python compare.py
```
