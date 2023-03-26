# Transformers:

Building visualization still in progress

1. Baseline for use a transformer model without fine-tuning for specifi tasks:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```
2. fine-tuning Bert for Question Answering: See notebook [QA_SQuAD](https://github.com/EthanWTL/Al-Models/blob/main/QA_SQuAD.ipynb)

![image](https://user-images.githubusercontent.com/97998419/226233347-061ec99b-7605-41da-aeb0-334203ae1385.png)


3.Bert+Bart: Long-form Question Answering:
https://yjernite.github.io/lfqa.html

![image](https://user-images.githubusercontent.com/97998419/226233267-736cc8ef-2987-413d-a7b4-58244da9ac2e.png)


4.persona chatbot
https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
