
from pathlib import Path
import shutil

# Update these paths
source_root = Path("/home/lakitha-senarathna/Downloads/newData")
output_root = Path("/home/lakitha-senarathna/Downloads/combinedData")

image_extensions = {
    ".jpg", ".jpeg", ".png"
}

# Identify source folders, excluding the output folder
source_folders = [
    folder for folder in source_root.iterdir()
    if folder.is_dir()
    and folder.resolve() != output_root.resolve()
]

output_root.mkdir(parents=True, exist_ok=True)

total_copied = 0
categories_found = set()

for source in source_folders:
    for category in source.iterdir():

        if not category.is_dir():
            continue

        categories_found.add(category.name)

        # Create a category even if it doesn't exist in other sources
        destination = output_root / category.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in category.iterdir():
            if (
                image_path.is_file()
                and image_path.suffix.lower() in image_extensions
            ):
                # Prevent collisions between identical filenames
                new_name = f"{source.name}_{image_path.name}"
                destination_file = destination / new_name

                shutil.copy2(image_path, destination_file)
                total_copied += 1

print("Dataset combination complete!")
print(f"Source folders processed: {len(source_folders)}")
print(f"Categories found: {len(categories_found)}")
print(f"Images copied: {total_copied}")
print(f"Output folder: {output_root}")

# Show image counts per category
for category in sorted(categories_found):
    folder = output_root / category
    count = sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    )
    print(f"{category}: {count} images")
