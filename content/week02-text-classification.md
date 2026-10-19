---
week: 2
title: Text Classification with Transformers
source: https://albanna-tutorials.com/textClassification.html
author: Dr. Abedal-Kareem Al-Banna (Dr. Abdulkarim Albanna), University of Petra
---
# Text Classification with Transformers

> *Build Your First NLP Model.* Level: Beginner NLP · Estimated time: ~60 minutes · Stack: Hugging Face, PyTorch, Transformers, scikit-learn. Material by Dr. Abedal-Kareem Al-Banna (ilmy / albanna-tutorials.com), following the approach of Tunstall, von Werra & Wolf, *NLP with Transformers*, Chapter 2.

## What You'll Learn

By the end of this tutorial you will be able to:

- Load and explore datasets from the Hugging Face Hub
- Tokenize and prepare text for a transformer
- Extract features from pre-trained models
- Build a simple baseline classifier
- Fine-tune a transformer model
- Deploy your model to the Hugging Face Hub

**Goal:** a working **emotion classification model** that identifies the emotion expressed in a piece of text.

**Prerequisites:** no prior knowledge of transformers is needed; basic Python and machine-learning familiarity helps.

## 1. What are Transformers?

Transformers are neural network architectures that changed NLP by using **attention**. Attention lets a model weigh the context of each word far more effectively than earlier approaches.

**Key features**

- **Attention mechanism:** the model focuses on the most relevant parts of the input.
- **Pre-training and fine-tuning:** the model first learns general language, then adapts to a specific task.
- **Parallelization:** the whole sequence is processed at once, unlike RNNs which go step by step.
- **Transfer learning:** knowledge learned from huge datasets carries over to your specific task.

**Popular transformer models**

- **BERT** — Bidirectional Encoder Representations from Transformers
- **DistilBERT** — a lighter, faster BERT variant with similar performance
- **RoBERTa** — a BERT variant with an optimized training recipe
- **GPT models** — Generative Pre-trained Transformers

### Setting up your environment

Required tools: Python 3.10+, PyTorch, the Hugging Face libraries, and scikit-learn for evaluation.

```python
!pip install transformers[torch]
!pip install datasets
!pip install scikit-learn
!pip install huggingface_hub
!pip install pandas
!pip install matplotlib
!pip install torch
```

> **Tip:** If installation fails, consult the library's installation guide. Google Colab is a good option because most of these libraries are pre-installed.

### Mini-challenge: verify your setup

Import `torch`, `transformers` and `datasets` and print each version to confirm everything imports successfully.

```python
import torch
import transformers
import datasets

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
```

## 2. Loading and Exploring the Dataset

The Hugging Face **Datasets** library gives easy access to many public datasets. This tutorial uses the **emotion** dataset (`dair-ai/emotion`): English tweets labeled with one of six emotions.

### Loading the dataset

```python
from datasets import load_dataset

emotions = load_dataset("dair-ai/emotion")
print(emotions)
```

The dataset has **train**, **validation** and **test** splits. Each example has a `text` field and an integer `label` from 0 to 5.

### Examining dataset structure

```python
emotions["train"].features
emotions["train"].column_names
emotions["train"]["text"][:2]
emotions["train"]["label"][:2]
```

Label names, in order: **sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)**.

### Converting to pandas

```python
import pandas as pd

emotions.set_format(type="pandas")
df = emotions["train"][:]

def convert_label_int_2_string(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(convert_label_int_2_string)
df.head()
```

### Visualizing the data

```python
import matplotlib.pyplot as plt

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Distribution of Emotions in Dataset")
plt.xlabel("Count")
plt.ylabel("Emotion")
plt.show()
```

This chart reveals the class distribution and any **class imbalance** (joy and sadness dominate; surprise and love are rare).

### Mini-challenge: class percentages

Compute the percentage of each emotion in the training set and identify the most and least common emotions. This tells you whether the model will need class-imbalance handling.

```python
percentages = df["label_name"].value_counts(normalize=True) * 100
print(percentages)
```

## 3. Tokenization and Data Preparation

**Tokenization** splits text into tokens and maps each token to a numeric ID so the model can process it.

### What is tokenization? (character-level example)

```python
text = "My name is kareem"
tokenized_text = list(text)
print(tokenized_text)

token2index = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2index)
# {' ': 0, 'M': 1, 'a': 2, 'e': 3, 'i': 4, 'k': 5, 'm': 6, 'n': 7, 'r': 8, 's': 9, 'y': 10}
```

Production systems use **subword** tokenization methods such as WordPiece (BERT), BPE (GPT) or SentencePiece.

### Using Hugging Face tokenizers

```python
from transformers import AutoTokenizer

model_ckp = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckp)

text = "My name is kareem"
encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))
```

### Tokenizer properties

```python
print(tokenizer.vocab_size)          # vocabulary size
print(tokenizer.model_input_names)   # inputs the model expects
print(tokenizer.model_max_length)    # maximum sequence length
```

The tokenizer output contains `input_ids` (token IDs), `attention_mask` (which tokens to attend to), and special tokens such as `[CLS]` and `[SEP]`.

### Tokenizing the entire dataset

```python
emotions.reset_format()   # undo the pandas format set earlier

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotion_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotion_encoded.column_names)
```

- `padding=True` pads to the longest sequence in the batch (`padding="max_length"` pads to the model's max length instead). Padding adds special tokens so sequences in a batch have equal length.
- `truncation=True` shortens sequences that exceed the model's maximum length. This prevents errors but may cut information.
- `batched=True` processes the data in batches; `batch_size=None` treats the whole split as one batch.
- New columns `input_ids` and `attention_mask` are added alongside the originals.

### Mini-challenge: tokenize a custom sentence

Tokenize *"I am feeling both happy and sad at the same time."*, print the IDs and tokens, and identify which words split into subword pieces. This is a useful debugging skill.

```python
sentence = "I am feeling both happy and sad at the same time."
encoded = tokenizer(sentence)
print(encoded.input_ids)
print(tokenizer.convert_ids_to_tokens(encoded.input_ids))
```

## 4. Feature Extraction from Pre-trained Models

Instead of fine-tuning, you can use a pre-trained model as a **frozen feature extractor** and train a simpler classifier on its outputs. This is faster and less resource-intensive.

### Loading the pre-trained model

```python
from transformers import AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_chk = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_chk).to(device)   # base model, no classification head
```

**Hidden states.** The model produces a contextual vector (hidden state) for every token. The final layer carries the most task-relevant information. The first token, `[CLS]`, is often used as a summary of the whole sentence, so we take its final-layer representation.

### Feature extraction function

```python
def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}
```

### Processing the dataset

```python
emotion_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotion_encoded.map(extract_hidden_states, batched=True)

import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_valid.shape}")
```

**Benefits:** faster than fine-tuning, less compute, still uses pre-trained knowledge, works well on small datasets.
**Limitations:** features are not tuned to the task, may miss task-specific nuances, and the performance ceiling is usually lower than fine-tuning.

### Mini-challenge: inspect the features

```python
print(f"Feature vector size: {X_train.shape[1]}")

labels = emotions["train"].features["label"].names
for i, emotion in enumerate(labels):
    mean_vec = X_train[y_train == i].mean(axis=0)
    print(f"{emotion}: {mean_vec[:5]} ...")

# Optional visualization
# !pip install umap-learn
# import umap
# reducer = umap.UMAP()
# X_2d = reducer.fit_transform(X_train)
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="viridis", s=2)
# plt.colorbar()
# plt.show()
```

The visualization shows whether the `[CLS]` features already separate the emotions.

## 5. Creating a Baseline Model with Logistic Regression

Always set a performance benchmark before fine-tuning.

### Dummy classifier

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_score = dummy_clf.score(X_valid, y_valid)
print(f"Dummy classifier accuracy: {dummy_score:.4f}")
```

### Logistic regression

```python
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_score = lr_clf.score(X_valid, y_valid)
print(f"Logistic regression accuracy: {lr_score:.4f}")
print(f"Improvement over dummy: {lr_score - dummy_score:.4f}")
```

### Confusion matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

y_preds = lr_clf.predict(X_valid)
labels = emotions["train"].features["label"].names
plot_confusion_matrix(y_preds, y_valid, labels)
```

The matrix shows which emotions are confused with one another.

### Additional metrics

```python
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_preds, target_names=labels))
```

**Why build a baseline?** It sets minimum expectations, is quick to build, reveals which classes are easy or hard, and gives a reference point. A suspiciously strong baseline may indicate dataset issues; similar scores across models point to feature limits; class imbalance usually shows up here.

### Mini-challenge: random forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_score = rf_clf.score(X_valid, y_valid)
print(f"Random forest accuracy: {rf_score:.4f}  (logistic regression: {lr_score:.4f})")

rf_preds = rf_clf.predict(X_valid)
rf_report = classification_report(y_valid, rf_preds, target_names=labels)
print(rf_report)
```

Which emotions does the random forest predict better?

## 6. Fine-tuning a Transformer Model

Fine-tuning updates **all** of the model's weights on the task, usually giving better results than frozen features.

### Loading a classification model

```python
from transformers import AutoModelForSequenceClassification

num_labels = 6
id2label = {i: name for i, name in enumerate(labels)}
label2id = {name: i for i, name in enumerate(labels)}
model = (AutoModelForSequenceClassification
         .from_pretrained(model_chk, num_labels=num_labels,
                          id2label=id2label, label2id=label2id)
         .to(device))
```

Same DistilBERT base as before, now with a randomly initialized classification head on top. Passing `id2label`/`label2id` stores the emotion names in the model config, so the deployed model reports `joy` rather than `LABEL_1`.

### Evaluation metrics

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

### Setting up the Trainer

```python
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotion_encoded["train"]) // batch_size
model_name = f"{model_chk}-finetuned-emotion"

training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True,
    log_level="error",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotion_encoded["train"],
    eval_dataset=emotion_encoded["validation"],
    processing_class=tokenizer,
)
```

| Argument | Meaning |
|---|---|
| `output_dir` | Where checkpoints and logs are saved |
| `num_train_epochs` | Number of passes over the training data |
| `learning_rate` | Step size for the optimizer (2e-5 is a common choice for BERT-style models) |
| `per_device_train_batch_size` | Examples per device per step |
| `weight_decay` | L2 regularization strength |
| `eval_strategy` | When to evaluate: `"epoch"` or `"steps"` |
| `push_to_hub` | Upload the model to the Hugging Face Hub during training |

> **Version note:** Transformers releases before 4.46 used the older names `evaluation_strategy` and `Trainer(tokenizer=...)`; the current names are `eval_strategy` and `processing_class=...`. `push_to_hub=True` requires that you are logged in (see Section 8).

### Training the model

```python
trainer.train()
```

Training runs for the configured epochs, reports metrics after each epoch, saves checkpoints and shows a progress bar.

### Mini-challenge: tuning experiments

```python
# 1. Try a different learning rate
training_args.learning_rate = 5e-5

# 2. Try a different scheduler
training_args.lr_scheduler_type = "cosine"

# 3. More epochs with early stopping
from transformers import EarlyStoppingCallback
training_args.num_train_epochs = 5
training_args.save_strategy = "epoch"        # must match eval_strategy when loading the best model
training_args.load_best_model_at_end = True
training_args.metric_for_best_model = "f1"   # required by EarlyStoppingCallback
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# Re-run trainer.train() after each change to see its effect
```

## 7. Evaluating Model Performance

### Evaluating on the validation set

```python
validation_results = trainer.evaluate()
print(f"Validation accuracy: {validation_results['eval_accuracy']:.4f}")
print(f"Validation F1: {validation_results['eval_f1']:.4f}")
print(f"Improvement over logistic regression: {validation_results['eval_accuracy'] - lr_score:.4f}")
```

### Confusion matrix

```python
preds_output = trainer.predict(emotion_encoded["validation"])
y_preds = preds_output.predictions.argmax(-1)
y_valid = preds_output.label_ids
plot_confusion_matrix(y_preds, y_valid, labels)
```

### Detailed classification report

```python
print(classification_report(y_valid, y_preds, target_names=labels))
```

Look for classes with low precision or recall.

### Error analysis

```python
validation_texts = emotions["validation"]["text"]
validation_labels = emotions["validation"]["label"]

errors = []
for i, (pred, true) in enumerate(zip(y_preds, y_valid)):
    if pred != true:
        errors.append({
            "text": validation_texts[i],
            "true": labels[true],
            "predicted": labels[pred],
        })

for i, err in enumerate(errors[:5]):
    print(f"{i+1}. Text: {err['text']}")
    print(f"   True: {err['true']}, Predicted: {err['predicted']}\n")
```

**Metric definitions:** accuracy = fraction correct; precision = of predicted positives, how many are right; recall = of the actual positives, how many were found; F1 = harmonic mean of precision and recall.

**Common error patterns:** confusion between similar emotions (love vs joy), ambiguous text expressing several emotions, sarcasm and figurative language, cultural or contextual references.

### Mini-challenge: predict on custom text

```python
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()
    return {
        "emotion": labels[predicted_class],
        "confidence": confidence,
        "all_probs": {labels[i]: probs[0][i].item() for i in range(len(labels))},
    }

test_texts = [
    "I can't believe I aced my exam!",
    "I miss my family so much right now.",
    "That driver just cut me off! So rude!",
]
for text in test_texts:
    result = predict_emotion(text)
    print(f"Text: {text}")
    print(f"Predicted: {result['emotion']} (confidence: {result['confidence']:.4f})")
    print(f"All probabilities: {result['all_probs']}\n")
```

## 8. Deploying the Model to Hugging Face Hub

### Setting up your Hugging Face account

```python
from huggingface_hub import notebook_login
notebook_login()
```

Create an access token at https://huggingface.co/settings/tokens. In a script (not a notebook) use `hf auth login` in the terminal instead (`huggingface-cli login` in older versions of `huggingface_hub`).

### Pushing the model to the Hub

```python
trainer.push_to_hub(commit_message="Training completed!")
```

This uploads the model weights, tokenizer configuration, model configuration and a README model card. The model appears at `https://huggingface.co/YOUR_USERNAME/distilbert-base-uncased-finetuned-emotion`.

### Creating a model card

```python
model_card = """
# Emotion Classification Model

This model classifies text into six emotions: sadness, joy, love, anger, fear, and surprise.

## Model Description
- Architecture: DistilBERT (distilbert-base-uncased)
- Fine-tuned on the DAIR.AI Emotion dataset
- Training accuracy: {train_acc:.4f}
- Validation accuracy: {val_acc:.4f}
- F1 score: {f1:.4f}

## Intended Usage
Sentiment analysis and emotion detection in English text.

## Limitations
- English only
- May struggle with sarcasm and ambiguous emotions
- May underperform on very short texts

## Training Procedure
- 2 epochs
- Learning rate: 2e-5
- Batch size: 64
""".format(
    train_acc=0.92,   # replace with your actual training accuracy
    val_acc=validation_results["eval_accuracy"],
    f1=validation_results["eval_f1"],
)

# from huggingface_hub import ModelCard
# ModelCard(model_card).push_to_hub("YOUR_USERNAME/distilbert-base-uncased-finetuned-emotion",
#                                   commit_message="Update model card")
```

### Using your model

```python
from transformers import pipeline

model_name = "YOUR_USERNAME/distilbert-base-uncased-finetuned-emotion"   # replace YOUR_USERNAME
classifier = pipeline("text-classification", model=model_name)

result = classifier("I'm so excited to see this working!")
print(result)   # e.g. [{'label': 'joy', 'score': 0.9874}]
```

**Hub benefits:** easy sharing, version control, automatic model cards, browser inference widgets, community discussions, download statistics, Spaces for demos, and framework integrations.

### Mini-challenge (optional): Gradio demo

```python
!pip install gradio
import gradio as gr

def predict(text):
    results = classifier(text, top_k=None)   # scores for all six emotions
    return {r["label"]: r["score"] for r in results}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter text here..."),
    outputs=gr.Label(num_top_classes=6),
    title="Emotion Classifier",
    description="Classifies text into six emotions: sadness, joy, love, anger, fear, surprise.",
)
demo.launch()
```

The demo can be hosted permanently on Hugging Face Spaces.

## 9. Next Steps and Conclusion

**What you learned:** loading datasets, tokenizing, extracting features, building scikit-learn baselines, fine-tuning for classification, evaluating with metrics and plots, and deploying to the Hub.

**Improving your model**

- **Different architectures:** RoBERTa, BERT, or others
- **Hyperparameter tuning:** grid or random search over learning rate, batch size, epochs
- **Data augmentation:** synonym replacement, random insertion, swapping
- **Class imbalance:** weighted loss, oversampling, synthetic examples

**Advanced topics:** prompt engineering, multi-task learning, few-shot learning, model distillation.

**Project ideas:** multi-label emotion detection, emotion-intensity prediction, a sentiment dashboard for social media or customer feedback, cross-lingual emotion detection.

**Resources**

- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/index
- Hugging Face Datasets docs: https://huggingface.co/docs/datasets/index
- scikit-learn docs: https://scikit-learn.org/stable/documentation.html
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course/chapter1/1
- DeepLearning.AI short courses: https://www.deeplearning.ai/short-courses/
- fast.ai: https://www.fast.ai/
- Hugging Face Forums: https://discuss.huggingface.co/
- r/MachineLearning: https://www.reddit.com/r/MachineLearning/
- Stack Overflow (huggingface-transformers tag): https://stackoverflow.com/questions/tagged/huggingface-transformers

Congratulations on building your first text classification model. Back to course: https://albanna-tutorials.com/cs/index.html · All courses: https://albanna-tutorials.com/courses.html

## Key concepts worth simulating

- **Subword tokenization:** type a sentence and watch WordPiece split it into `[CLS]`, tokens, `##pieces`, `[SEP]` with their IDs; highlight padding tokens and the attention mask when batching sentences of different length.
- **Padding and truncation:** show a batch of 3 sentences padded to the longest (`padding=True`) vs. to `max_length`, and what truncation chops off.
- **[CLS] feature extraction:** animate hidden states for every token flowing out of the last DistilBERT layer, then the `[CLS]` 768-d vector being pulled out; project the vectors of many tweets to 2D (UMAP-style) colored by emotion.
- **Feature extraction vs. fine-tuning:** frozen encoder + logistic regression vs. gradients flowing back through all layers; show which weights change.
- **Class imbalance:** interactive bar chart of the 6 emotion counts; show how a "most frequent" dummy classifier's accuracy equals the majority-class share.
- **Normalized confusion matrix:** click a cell to see example texts; illustrate love↔joy and fear↔surprise confusions.
- **Precision / recall / F1 per class:** slider on a decision threshold showing the trade-off and the harmonic mean.
- **Softmax over logits → predicted emotion and confidence:** vary 6 logits with sliders and watch the probability bars.
- **Training loop:** epochs × steps with learning rate 2e-5, loss curve, evaluation at each epoch; optional cosine schedule and early stopping.

## Test code

```python
# 1. Dataset + tokenizer sanity check (no training)
from datasets import load_dataset
from transformers import AutoTokenizer

emotions = load_dataset("dair-ai/emotion")
labels = emotions["train"].features["label"].names
print(emotions)
print(labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tokenizer("I am feeling both happy and sad at the same time.")
print(enc.input_ids)
print(tokenizer.convert_ids_to_tokens(enc.input_ids))
```

```python
# 2. [CLS] feature extraction + logistic-regression baseline on a small subset
import numpy as np, torch
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

def cls_features(texts, bs=32):
    feats = []
    for i in range(0, len(texts), bs):
        batch = tokenizer(texts[i:i+bs], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats.append(model(**batch).last_hidden_state[:, 0].cpu().numpy())
    return np.vstack(feats)

train = emotions["train"].shuffle(seed=0).select(range(1000))
valid = emotions["validation"].select(range(300))
X_train, y_train = cls_features(train["text"]), np.array(train["label"])
X_valid, y_valid = cls_features(valid["text"]), np.array(valid["label"])
print("feature shape:", X_train.shape)

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
lr = LogisticRegression(max_iter=3000).fit(X_train, y_train)
print(f"dummy: {dummy.score(X_valid, y_valid):.3f}   logistic regression: {lr.score(X_valid, y_valid):.3f}")
```

```python
# 3. Quick fine-tune on a subset (1 epoch) and evaluate
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

small = emotions["train"].shuffle(seed=0).select(range(2000)).map(tokenize, batched=True)
small_val = emotions["validation"].select(range(500)).map(tokenize, batched=True)

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds),
            "f1": f1_score(pred.label_ids, preds, average="weighted")}

clf = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6).to(device)
args = TrainingArguments(output_dir="distilbert-emotion-test", num_train_epochs=1, learning_rate=2e-5,
                         per_device_train_batch_size=32, per_device_eval_batch_size=32,
                         weight_decay=0.01, eval_strategy="epoch", report_to="none", log_level="error")
trainer = Trainer(model=clf, args=args, compute_metrics=compute_metrics,
                  train_dataset=small, eval_dataset=small_val, processing_class=tokenizer)
trainer.train()
print(trainer.evaluate())
```

```python
# 4. Predict emotion with confidence from the fine-tuned model
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(clf(**inputs).logits, dim=-1)[0]
    idx = int(probs.argmax())
    return labels[idx], float(probs[idx]), {l: round(float(p), 3) for l, p in zip(labels, probs)}

for t in ["I can't believe I aced my exam!",
          "I miss my family so much right now.",
          "That driver just cut me off! So rude!"]:
    print(t, "->", predict_emotion(t))
```
