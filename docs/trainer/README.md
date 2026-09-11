# Model Training Pipeline: Corpus, Precompute, Shards, Teacher & Student

## 1. High-Level Overview

A useful way to think about modern LLM training is:

```text
Raw Data
   ↓
Corpus
   ↓
Tokenization / Cleaning
   ↓
Train / Validation Split
   ↓
Sharding
   ↓
Precompute (Val)
   ↓
Precompute (Train)
   ↓
Actual Training
   ↓
Checkpoints
```

If teacher/student distillation is involved, the pipeline may instead look like:

```text
Corpus
   ↓
Tokenization
   ↓
Train / Validation Split
   ↓
Teacher Inference
   ↓
Precompute Teacher Outputs
   ↓
Shards
   ↓
Student Training
   ↓
Checkpoints
```

---

# 2. Corpus

The **corpus** is the collection of data used for training.

For an LLM, this might include:

- Web pages
- Books
- Wikipedia
- Source code
- Scientific papers
- Conversations
- Synthetic data
- Domain-specific datasets

For example:

```text
corpus/
├── web/
│   ├── web_0001.jsonl
│   ├── web_0002.jsonl
│   └── ...
├── code/
├── books/
├── wikipedia/
└── synthetic/
```

A raw record could look like:

```json
{
    "text": "PyTorch is a machine learning framework..."
}
```

Corpus preparation usually involves:

```text
Raw Documents
      ↓
Cleaning
      ↓
Language Detection
      ↓
Quality Filtering
      ↓
Deduplication
      ↓
Safety / Content Filtering
      ↓
Final Corpus
```

When someone says:

> "We trained on a 2T-token corpus."

they usually mean the cleaned training corpus contains approximately **2 trillion tokenizer tokens**.

---

# 3. Tokenization

The neural network does not directly operate on strings.

Text:

```text
"The cat sat on the mat."
```

is converted by a tokenizer into integer IDs:

```text
"The cat sat on the mat."

        ↓ tokenizer

[1820, 8412, 4251, 402, 279, 5634, 13]
```

These integers are the actual inputs consumed by the model.

Conceptually:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Model
```

---

# 4. Train / Validation Split

The corpus is normally divided into at least:

```text
Corpus
│
├── Training Data
│
└── Validation Data
```

For example:

```text
100B tokens total

99.9B → training
 0.1B → validation
```

The critical difference is:

```text
Training data
    ↓
changes model weights

Validation data
    ↓
measures model quality
but DOES NOT change weights
```

---

# 5. What Does "Precompute" Mean?

`precompute` means doing expensive deterministic work **before actual training** and saving the result.

Instead of repeatedly doing:

```text
Read JSON
 ↓
Parse text
 ↓
Tokenize
 ↓
Pack sequences
 ↓
Prepare metadata
 ↓
Train
```

every training step, we can do it once:

```text
Corpus
 ↓
Precompute
 ↓
Training-ready binary data
 ↓
Training
```

Precomputation may include:

- Tokenization
- Sequence packing
- Document boundaries
- Sample indexes
- Attention information
- Position information
- Dataset mixing information
- Teacher outputs
- Embeddings
- Image latents
- Text embeddings

The exact meaning depends on the training framework.

---

# 6. Precompute (Train)

`precompute(train)` prepares the training dataset.

For example:

```text
training corpus
      ↓
tokenize
      ↓
pack sequences
      ↓
create indexes
      ↓
write shards
```

Output might look like:

```text
train/
├── shard_00000.bin
├── shard_00000.idx
├── shard_00001.bin
├── shard_00001.idx
├── shard_00002.bin
├── shard_00002.idx
└── ...
```

These files are then consumed by the actual training process.

---

# 7. Precompute (Val)

Validation data is prepared similarly:

```text
validation corpus
       ↓
precompute(val)
       ↓
validation shards
```

For example:

```text
val/
├── shard_00000.bin
├── shard_00000.idx
└── ...
```

During validation:

```python
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch)
        val_loss += calculate_loss(logits)
```

There is no:

```python
optimizer.step()
```

Therefore validation does **not** modify the model.

---

# 8. Why Precompute Validation First?

You may see:

```text
corpus
 ↓
precompute(val)
 ↓
precompute(train)
 ↓
train
```

Validation is normally much smaller than training.

For example:

```text
Training:    5,000,000,000,000 tokens
Validation:        100,000,000 tokens
```

Preparing validation first can verify that:

- Tokenization works
- Dataset format works
- Sequence packing works
- Model inputs work
- Evaluation works
- Teacher inference works

before spending hours or days processing the full training corpus.

---

# 9. What Is a Shard?

A **shard is simply one piece of a larger dataset**.

Suppose you have:

```text
1 trillion tokens
```

Instead of creating one enormous file:

```text
training.bin
```

you divide it into:

```text
shard_00000.bin
shard_00001.bin
shard_00002.bin
...
shard_09999.bin
```

For example:

```text
10,000 shards
×
100M tokens/shard

≈ 1 trillion tokens
```

Together, all shards represent the full dataset.

---

# 10. Why Shards?

Sharding helps with:

- Distributed training
- Parallel preprocessing
- Disk I/O
- Network storage
- Randomization
- Memory mapping
- Caching
- Failure recovery
- Multi-node training

For example:

```text
Dataset
   │
   ├── shard 001
   ├── shard 002
   ├── shard 003
   ├── shard 004
   └── ...
```

Different workers can read different parts:

```text
GPU Worker 0 ──→ shard 104
GPU Worker 1 ──→ shard 392
GPU Worker 2 ──→ shard 887
GPU Worker 3 ──→ shard 021
...
```

The exact assignment is normally controlled by the distributed sampler/data loader.

---

# 11. Shard vs Sample vs Batch

These are different concepts.

## Sample

A sample is one training sequence.

For context length 8192:

```text
sample 0 = 8192 tokens
sample 1 = 8192 tokens
sample 2 = 8192 tokens
```

---

## Batch

A batch contains multiple samples:

```text
Batch

sample 192
sample 814
sample 527
sample 104
```

Tensor shape might be:

```text
[4, 8192]
```

---

## Shard

A shard contains many samples:

```text
shard_005.bin

├── sample 0
├── sample 1
├── sample 2
├── ...
└── sample 12000
```

Therefore:

```text
CORPUS
  │
  ├── SHARD
  │     ├── sample
  │     ├── sample
  │     └── ...
  │
  ├── SHARD
  │     ├── sample
  │     └── ...
  │
  └── ...
```

Training takes **batches of samples** from those shards.

---

# 12. Actual Training

For causal language modeling, suppose we have:

```text
[BOS, The, cat, sat, on, the, mat, EOS]
```

The training problem is shifted by one token:

```text
INPUT                  TARGET

BOS                    The
The                    cat
cat                    sat
sat                    on
on                     the
the                    mat
mat                    EOS
```

Mathematically:

```text
Input:
[t0, t1, t2, t3, t4]

Target:
[t1, t2, t3, t4, t5]
```

The model tries to predict the next token.

A simplified training loop:

```python
for batch in training_data:

    optimizer.zero_grad()

    # Forward
    logits = model(batch.input_ids)

    # Calculate error
    loss = cross_entropy(
        logits,
        batch.labels
    )

    # Compute gradients
    loss.backward()

    # Update model weights
    optimizer.step()
```

The important distinction is:

```text
Corpus / Precompute
        ↓
prepares data


Training
        ↓
changes model weights
```

---

# 13. Training vs Validation

During training:

```text
Forward
 ↓
Loss
 ↓
Backward
 ↓
Optimizer Step
 ↓
Weights Changed
```

During validation:

```text
Forward
 ↓
Loss
 ↓
Report Metric
```

There is no backward pass and no optimizer update.

A typical run might look like:

```text
Step 1000
train loss = 3.41
val loss   = 3.55

Step 5000
train loss = 2.91
val loss   = 3.02

Step 10000
train loss = 2.53
val loss   = 2.70
```

---

# 14. Teacher and Student Models

Teacher/student training usually refers to **knowledge distillation**.

The basic idea is:

```text
Teacher = strong model
Student = model being trained
```

For example:

```text
Teacher
70B parameters
      │
      │ teaches
      ▼
Student
3B parameters
```

The goal is to transfer some of the teacher's capabilities into a smaller model.

---

# 15. Which Model Actually Gets Trained?

Usually:

```text
Teacher
   │
   └── FROZEN
       no optimizer update


Student
   │
   └── TRAINABLE
       backward()
       optimizer.step()
```

So:

> The teacher provides information. The student's weights are updated.

---

# 16. Normal Training vs Teacher Training

Normal training:

```text
Corpus
   ↓
Input
   ↓
Student
   ↓
Prediction
   ↓
Compare with ground truth
   ↓
Loss
   ↓
Backward
   ↓
Update Student
```

Teacher/student training:

```text
                 Input
                   │
         ┌─────────┴─────────┐
         ↓                   ↓
      Teacher             Student
         │                   │
         ↓                   ↓
Teacher Prediction     Student Prediction
         │                   │
         └─────────┬─────────┘
                   ↓
                  Loss
                   ↓
                Backward
                   ↓
             Update Student
```

The teacher normally remains frozen.

---

# 17. Hard Labels

Traditional language-model training uses the actual next token.

For example:

```text
Input:

"The capital of France is"
```

Ground truth:

```text
Paris
```

Conceptually:

```text
Paris       1.0
London      0.0
Lyon        0.0
Berlin      0.0
Rome        0.0
```

This is a **hard target**.

---

# 18. Soft Teacher Targets

A teacher can provide an entire probability distribution:

```text
Paris       0.82
Lyon        0.07
London      0.04
Rome        0.02
Berlin      0.01
...
```

The student might predict:

```text
Paris       0.45
London      0.25
Lyon        0.08
Rome        0.05
Berlin      0.03
...
```

Training can push the student's distribution toward the teacher's distribution.

These are called **soft targets**.

---

# 19. Distillation Loss

Traditional loss:

```python
hard_loss = cross_entropy(
    student_logits,
    ground_truth
)
```

Teacher/student training can add:

```python
soft_loss = KL_divergence(
    student_distribution,
    teacher_distribution
)
```

And combine them:

```python
loss = hard_loss + alpha * soft_loss
```

Conceptually:

```text
                 Total Loss
                     │
          ┌──────────┴──────────┐
          │                     │
    Ground Truth Loss      Teacher Loss
          │                     │
          ▼                     ▼
"What is actually      "What does the
 correct?"              teacher believe?"
```

---

# 20. Temperature

Knowledge distillation frequently uses a parameter called **temperature**:

```text
T
```

Normally:

```python
teacher_probs = softmax(teacher_logits)
```

With temperature:

```python
teacher_probs = softmax(
    teacher_logits / T
)
```

For example, with `T = 1`:

```text
Paris      0.95
London     0.02
Lyon       0.01
Rome       0.005
...
```

With a higher temperature:

```text
Paris      0.60
London     0.12
Lyon       0.10
Rome       0.07
...
```

The softer distribution exposes more information about how the teacher ranks alternative predictions.

---

# 21. Why Precompute the Teacher?

Suppose:

```text
Teacher = 70B
Student = 3B
```

Without precomputation, every training batch requires:

```text
Teacher 70B forward
        +
Student 3B forward
        +
Student 3B backward
```

That is expensive.

Instead, run the teacher once:

```text
             PRECOMPUTE

Corpus
   ↓
Teacher 70B
   ↓
Teacher Predictions
   ↓
Save to Disk
```

Then actual training becomes:

```text
                 TRAINING

Corpus ───────────────→ Student
                           │
Teacher Cache ─────────────┤
                           ↓
                          Loss
                           ↓
                        Backward
                           ↓
                    Update Student
```

The teacher doesn't even have to be loaded during student training.

---

# 22. Teacher Precompute Shards

For a large dataset, teacher outputs can also be sharded:

```text
teacher_cache/
├── shard_00000
├── shard_00001
├── shard_00002
├── ...
└── shard_00999
```

A shard could conceptually contain:

```text
SHARD 001

Sequence #1
------------------
input tokens
teacher predictions

Sequence #2
------------------
input tokens
teacher predictions

Sequence #3
------------------
input tokens
teacher predictions
```

The student training process reads both the original input and the cached teacher information.

---

# 23. Full Teacher Logits Are Huge

Suppose:

```text
sequence length = 4096
vocabulary      = 128,000
```

The teacher produces:

```text
128,000 logits
```

for every token.

Therefore one sequence has:

```text
4096 × 128,000

≈ 524 million logits
```

At FP16:

```text
524M × 2 bytes

≈ 1 GB
```

for just one sequence.

This makes storing complete teacher distributions extremely expensive.

---

# 24. Top-K Teacher Logits

A common solution is storing only the teacher's most important predictions.

Instead of:

```text
128,000 logits / token
```

store:

```text
Top 8
Top 16
Top 32
Top 64
```

For example:

```text
Teacher predictions:

Token ID      Probability
--------------------------
8291          0.643
928           0.142
17321         0.071
521           0.043
77            0.021
...
```

A precomputed record could therefore contain:

```text
input_tokens

top_k_token_ids

top_k_logits
```

For example:

```python
{
    "tokens": [...],

    "teacher_topk_ids": [
        [8291, 928, 17321, 521],
        ...
    ],

    "teacher_topk_logits": [
        [8.72, 7.21, 6.51, 6.02],
        ...
    ]
}
```

This can dramatically reduce storage.

---

# 25. Response Distillation

There is another important type of teacher/student training.

Instead of saving logits, the teacher generates complete answers.

For example:

```text
Prompt:

"Explain why the sky is blue."

        ↓

Teacher

        ↓

"Sunlight contains many wavelengths..."
```

You then save:

```json
{
    "prompt": "Explain why the sky is blue.",
    "response": "Sunlight contains many wavelengths..."
}
```

The student trains on that generated corpus.

The pipeline becomes:

```text
Prompts
   ↓
Teacher
   ↓
Synthetic Responses
   ↓
Synthetic Corpus
   ↓
Student Training
```

This is often called **response distillation** or **synthetic-data distillation**.

---

# 26. Three Important Training Methods

## Normal Pretraining

```text
Real Corpus
    ↓
Student
    ↓
Cross Entropy
    ↓
Update Student
```

The teacher is not involved.

---

## Response Distillation

```text
Prompts
   ↓
Teacher
   ↓
Generated Responses
   ↓
Student
   ↓
Cross Entropy
   ↓
Update Student
```

The teacher generates the training dataset.

---

## Logit Distillation

```text
                 Input
                   │
         ┌─────────┴─────────┐
         ↓                   ↓
      Teacher             Student
         │                   │
 Teacher logits        Student logits
         │                   │
         └─────────┬─────────┘
                   ↓
            Distillation Loss
                   ↓
                Backward
                   ↓
             Update Student
```

The student attempts to reproduce the teacher's probability distribution.

---

# 27. Teacher Does Not Necessarily Have to Be Larger

Usually:

```text
Teacher > Student
```

For example:

```text
Teacher = 70B
Student = 8B
```

But this isn't required.

A teacher can be:

- A larger model
- A better-trained model
- A reasoning model
- An ensemble
- A specialized model
- A model using retrieval
- A model using tools
- A model using longer inference
- A model given extra context

For example:

```text
Teacher

8B model
+
Retrieval
+
Tools
+
Long reasoning

        ↓ teaches

Student

8B model
without those extras
```

The distinction is primarily about **role**, not parameter count.

---

# 28. Complete Teacher/Student Training Pipeline

Putting everything together:

```text
                         RAW DATA
                            │
                            ▼
                         CORPUS
                            │
                            ▼
                    CLEAN / FILTER
                            │
                            ▼
                       TOKENIZE
                            │
                            ▼
                    TRAIN / VAL SPLIT
                            │
              ┌─────────────┴─────────────┐
              │                           │
            TRAIN                        VAL
              │                           │
              ▼                           ▼
        Teacher Model               Teacher Model
          Inference                   Inference
              │                           │
              ▼                           ▼
      Precompute(train)           Precompute(val)
              │                           │
              ▼                           ▼
         Train Shards                 Val Shards
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                       Data Loader
                            │
                            ▼
                       STUDENT MODEL
                            │
                            ▼
                       Forward Pass
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Ground Truth Loss           Teacher Loss
              │                           │
              └─────────────┬─────────────┘
                            ▼
                         Total Loss
                            │
                            ▼
                         Backward
                            │
                            ▼
                     Optimizer Step
                            │
                            ▼
                    UPDATED STUDENT
                            │
                            ▼
                       Checkpoint
```

---

# 29. Which Stages Use GPUs?

A simplified view:

| Stage | CPU | GPU | Changes Weights? |
|---|---:|---:|---:|
| Corpus collection | ✓ | Usually no | No |
| Cleaning/filtering | ✓ | Sometimes | No |
| Tokenization | ✓ | Sometimes | No |
| Sharding | ✓ | Usually no | No |
| Teacher precompute | ✓ | **✓** | No |
| Validation | ✓ | **✓** | No |
| Student forward | ✓ | **✓** | No |
| Student backward | ✓ | **✓** | No |
| Optimizer step | ✓ | **✓** | **Yes** |
| Checkpoint save | ✓ | Sometimes | No |

The critical moment where learning actually occurs is:

```text
loss.backward()
       ↓
gradients
       ↓
optimizer.step()
       ↓
MODEL WEIGHTS CHANGE
```

---

# 30. Simple Mental Model

The easiest way to remember everything is:

```text
CORPUS
"What data do I have?"

        ↓

PRECOMPUTE
"Turn it into something training can consume efficiently."

        ↓

SHARDS
"Split that enormous dataset into manageable pieces."

        ↓

TEACHER
"What should the student learn?"

        ↓

STUDENT
"The model whose weights we actually want."

        ↓

BATCH
"What subset of samples are the GPUs processing right now?"

        ↓

FORWARD
"What does the student predict?"

        ↓

LOSS
"How wrong is the student?"

        ↓

BACKWARD
"Which weights caused the error?"

        ↓

OPTIMIZER
"How should those weights change?"

        ↓

CHECKPOINT
"Save what the student has learned."
```

---

# 31. One-Sentence Definitions

**Corpus:** The complete collection of source training data.

**Token:** The numerical unit of text consumed by the model.

**Sample / Sequence:** One model input, often containing thousands of tokens.

**Shard:** A storage-sized piece of the larger dataset containing many samples.

**Batch:** A collection of samples processed together during one training iteration.

**Precompute:** Work performed ahead of training so it does not have to be repeatedly calculated.

**Teacher:** A model that provides richer training targets or generated examples.

**Student:** The model whose parameters are being learned.

**Forward pass:** Run input through the model to obtain predictions.

**Loss:** A numerical measurement of how wrong the model is.

**Backward pass:** Compute gradients indicating how parameters contributed to the loss.

**Optimizer step:** Modify model parameters using those gradients.

**Validation:** Measure model performance without changing model weights.

**Checkpoint:** A saved snapshot of model weights and usually training state.

---

# 32. Final Picture

```text
                         DATA SIDE
                            │
                            ▼
                    ┌──────────────┐
                    │    CORPUS    │
                    └──────┬───────┘
                           │
                     tokenize/filter
                           │
                           ▼
                    ┌──────────────┐
                    │    SHARDS    │
                    └──────┬───────┘
                           │
                           ▼

                     TEACHER SIDE
                           │
                           ▼
                    ┌──────────────┐
                    │   TEACHER    │
                    │    MODEL     │
                    └──────┬───────┘
                           │
                        inference
                           │
                           ▼
                    ┌──────────────┐
                    │ PRECOMPUTED  │
                    │   TARGETS    │
                    └──────┬───────┘
                           │
                           ▼

                     STUDENT SIDE
                           │
                           ▼
                    ┌──────────────┐
                    │   STUDENT    │
                    │    MODEL     │
                    └──────┬───────┘
                           │
                        forward
                           │
                           ▼
                    ┌──────────────┐
                    │     LOSS     │
                    └──────┬───────┘
                           │
                        backward
                           │
                           ▼
                    ┌──────────────┐
                    │  OPTIMIZER   │
                    │    STEP      │
                    └──────┬───────┘
                           │
                    weights updated
                           │
                           ▼
                    ┌──────────────┐
                    │  CHECKPOINT  │
                    └──────────────┘
```

The most important distinction is:

> **Corpus/shards/precompute are primarily about preparing and delivering training information. Teacher inference produces additional supervision. The student is the model being learned. The actual learning happens when the loss is backpropagated and the optimizer updates the student's weights.**