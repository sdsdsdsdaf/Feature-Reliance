from datasets import load_dataset


train_ds = load_dataset(
    "ILSVRC/imagenet-1k",
    split="train"
)

train_ds.save_to_disk("/home/user/projects/Feature-Reliance")
