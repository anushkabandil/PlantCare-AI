import os

train_dir = "dataset/New Plant Diseases Dataset(Augmented)/train"

classes = os.listdir(train_dir)

print("Total classes:", len(classes))
print("Some classes:", classes[:10])