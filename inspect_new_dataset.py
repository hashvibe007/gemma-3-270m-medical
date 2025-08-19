from datasets import load_dataset

dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", streaming=True)

print("Dataset features:")
# For streaming datasets, features are not available directly.
# We can inspect the first element to see the structure.
example = next(iter(dataset))
print(example.keys())

print("\nFirst 5 examples:")
for i, example in enumerate(dataset):
    if i >= 5:
        break
    print(example)


