import shutil
from pathlib import Path
from tqdm import tqdm

base_path = Path("/home/ai_center/ai_users/roeibenzion/VLM-FGA/LLaVA_converter/playground/data/images")

def fix_nested_dirs(root: Path):
    for parent in root.glob("**/*"):
        if not parent.is_dir():
            continue

        nested = parent / parent.name
        if not nested.is_dir():
            continue

        try:
            items = list(nested.iterdir())
        except Exception as e:
            print(f"  Skipping {nested}: {e}")
            continue

        if not items:
            continue

        print(f"\nFlattening: {nested} ? {parent}")
        for item in tqdm(items, desc=f"Moving from {nested.name}", unit="file"):
            dest = parent / item.name
            if dest.exists():
                continue
            try:
                shutil.move(str(item), str(dest))
            except Exception as e:
                print(f"  Failed to move {item.name}: {e}")

        print(f"  ? Flattened, kept: {nested}")

if __name__ == "__main__":
    fix_nested_dirs(base_path)
