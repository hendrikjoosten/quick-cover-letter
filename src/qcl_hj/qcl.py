from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys

file_name = sys.argv[1]

with open(file_name) as f:
    lines = f.readlines()

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

inputs = tokenizer("summerize" + str(lines), return_tensors="pt")

outputs = model.generate(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))