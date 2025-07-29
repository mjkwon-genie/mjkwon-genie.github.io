---
layout: post
title: "Decoder-Only Architecture Overview"
---

1. **Model Input and Preprocessing**

- **Tokenization**
  Convert the raw text into a sequence of token IDs using a tokenizer.
  
  Example: "Hello I'm Jenna" → [101, 455, 999, 203]

- **Token Embedding**
  An embedding table of shape (vocab_size, m).
  
  Each token ID is looked up in the embedding table to create an (n, m) matrix.
  This lookup is done via indexing, not matrix multiplication, making it fast and efficient.

- **Positional Encoding**
  Create an (n, m) matrix containing positional information for each token.
  
  *Sinusoidal*: Use sin and cos formulas per position and dimension (as in the Transformer paper).
  
  *Learned*: Slice n positions from a (max_position, m) table.
  
  Add the token embeddings and positional encodings to form the final (n, m) input matrix.

2. **Decoder Block (Repeated N times)**

A. **Pre-Norm (LayerNorm/RMSNorm)**
  Apply RMSNorm or LayerNorm before attention/FFN.
  RMSNorm omits the mean calculation for simplicity and speed (used in Llama models).

B. **Grouped Query Masked Self-Attention**
  Use many query heads (h) but fewer key/value heads (h_g), e.g., 32:8 or 8:1~4:1.
  Split the input (n, m) per head:
  
  Q: h tensors of shape (n, m/h)
  
  K/V: h_g tensors of shape (n, m/h_g)
  
  *Masked Attention* ensures each token only attends to previous positions (future positions are masked as -inf).
  
  Softmax is applied row-wise so each query's probabilities sum to 1.
  
  *Flash Attention* processes the (n, n) matrix in blocks to maximize memory and speed efficiency, allowing long contexts without exploding compute.
  
  Dropout can be applied at various points to prevent overfitting and improve generalization.

C. **LoRA Adapter (Optional)**
  Attach low-rank adapters to major linear weights like Wq and Wv (optionally Wk, FFN linear layers).
  
  For each head, add an A matrix of shape (m, r) and a B matrix of shape (r, m/h).
  
  During training, the original weights stay fixed and only the low-rank A/B matrices are learned.

D. **Residual Connection**
  Add the input to the output of attention/FFN to preserve information and stabilize gradient flow:
  output = f(x) + x
  
  This alleviates vanishing/exploding gradients and enables deep networks to train reliably.

E. **Feed Forward Network (FFN)**
  Structure: Linear(m, d_ff) → Activation(GeLU) → Dropout → Linear(d_ff, m) → Dropout
  
  GeLU provides smooth nonlinearity and works seamlessly with automatic differentiation.
  
  LoRA can also be applied to the first/second linear layers of the FFN.

F. **(Residual, Norm, Dropout Repeats)**
  Each module is followed by residual + norm + dropout for generalization, stable training, and smooth information flow.

3. **Final Output / Generation Stage**

- **Pass Through N Decoder Blocks**
  Obtain an (n, m) output matrix.

- **Final Linear Layer (Fully Connected)**
  Map (n, m) → (n, vocab_size).
  Each token position receives logits for the next possible token.
  Weight tying with the embedding table is common.

- **Softmax & Probability Interpretation**
  Apply softmax per row to get the next-token probabilities.
  During generation, take the last row and select the next token via argmax or sampling.

- **Autoregressive Generation**
  Generate one token at a time, repeating for n steps.
  At each step, append the new token to the input and predict again (chain method).

4. **Training Process / Loss Function**

- **Loss Function**
  Use Cross Entropy Loss, which is a special case of Negative Log Likelihood.
  Compare the model output (n, vocab_size) with the ground truth token IDs (n,).
  Internally, softmax, log, and NLL are all handled automatically.
  Update parameters so that the probability of the correct token at each position approaches 1.

- **Backpropagation**
  Every operation in the computation graph (linear, activation, normalization, etc.) is a node.
  Gradients are automatically computed for each parameter via the chain rule.
  Activation functions are also nodes and require gradients.

- **Optimizer (AdamW)**
  A base learning rate is specified (e.g., 2e-5, 3e-4).
  For each parameter, adaptive learning rate, momentum, and weight decay are computed dynamically.
  The optimizer combines gradient, base lr, previous state, and weight magnitude to update parameters efficiently and stably.

5. **Dropout, Residual, and Other Key Techniques**

- **Dropout**
  Applied in attention, FFN, and other modules.
  Dropout positions and rates are hyperparameters.
  Randomly disabling neurons during training helps prevent overfitting and leads to robust models.

- **Residual Connection**
  Add the block's input to its output.
  This preserves information, mitigates gradient vanishing/exploding, and enables stable training of deep networks.

- **LoRA**
  Keep original weights intact and train only low-rank matrices (A/B).
  This greatly reduces parameters, memory, and compute while enabling fast, effective fine-tuning.
  Usually applied to Wq and Wv, but can extend to K/FFN as needed.

- **Grouped Query Attention & Flash Attention**
  GQA increases the number of query heads relative to key/value heads to handle long contexts efficiently.
  Flash Attention performs attention in blocks for dramatic memory and speed improvements.
  Both are used together in modern LLMs.

6. **Overall Flow (Architecture Summary)**

Input: Raw sentence → Tokenizer → (n,) token IDs

Embedding: (n, m) embeddings + positional encoding

N Decoder Blocks:
  Norm → Grouped Query Masked Flash Attention (+Dropout, +LoRA) → Residual
  Norm → FFN (+Activation, +Dropout, +LoRA) → Residual
  Dropout/Residual/Norm/LoRA are combined effectively

Output: (n, m) → Linear → (n, vocab_size)

Softmax: Next-token probabilities at each position

Autoregressive generation produces one token at a time

Training: CrossEntropyLoss (softmax + NLL) and AdamW optimizer update parameters

7. **Example Implementation Flow (PyTorch)**

```python
# Forward pass
input_ids = tokenizer.encode("Hello I'm Jenna", return_tensors='pt')
emb = model.embedding(input_ids)  # (n, m)
pos_emb = model.positional_encoding(torch.arange(n))  # (n, m)
x = emb + pos_emb

for block in model.decoder_blocks:
    x = block(x)  # contains Norm, GQA+Flash, Residual, FFN, Dropout, LoRA

logits = model.final_linear(x)  # (n, vocab_size)
loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
# Optimizer step (AdamW, etc.)
```

8. **Final Summary**

The decoder-only LLM architecture follows a typical pattern:
  token embeddings + positional encoding → N decoder blocks (Flash GQA, FFN, Residual, Dropout, LoRA, etc.) → linear + softmax → CrossEntropyLoss + AdamW training → autoregressive token generation.

The interplay of these modules is the foundation of performance, efficiency, and scalability in modern deep-learning LLMs.
