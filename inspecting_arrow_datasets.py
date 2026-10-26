from datasets import load_from_disk

dataset = load_from_disk("dataset_with_image/connectivity")
print(dataset[0])