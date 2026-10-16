from pathlib import Path

data_dir = Path("/home/lakitha-senarathna/Downloads/combinedData")

image_extensions = {".jpg"}

class_counts = {}

for class_dir in sorted(data_dir.iterdir()):
    if class_dir.is_dir():
        count = sum(
            1 for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        )
        class_counts[class_dir.name] = count

for class_name, count in sorted(
    class_counts.items(), key=lambda item: item[1]
):
    print(f"{class_name}: {count}")

print("Total images:", sum(class_counts.values()))
print("Number of classes:", len(class_counts))