import os
from datasets import load_dataset

def download_levir_cdplus(output_dir, structure="T1T2GT", dataset_name="LEVIR-CD+"):
    # 若未显式指定镜像，默认走 HuggingFace 国内镜像以提高成功率
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    ds = load_dataset("blanchon/LEVIR_CDPlus")
    # 自动匹配列名，兼容不同版本字段
    cols = set(ds["train"].column_names)
    if {"image1", "image2", "mask"} <= cols:
        key_a, key_b, key_m = "image1", "image2", "mask"
    elif {"source_img", "target_img", "label_img"} <= cols:
        key_a, key_b, key_m = "source_img", "target_img", "label_img"
    else:
        raise ValueError(f"Unexpected columns {cols}, please set key mapping manually.")

    dirs_map = ("T1", "T2", "GT") if structure == "T1T2GT" else ("A", "B", "label")
    dest_root = os.path.join(output_dir, dataset_name)
    for split in ["train", "test", "val"]:
        for sub in dirs_map:
            os.makedirs(os.path.join(dest_root, split, sub), exist_ok=True)

    def save_split(split_key, split_name):
        if split_key not in ds:
            return False
        split = ds[split_key]
        names = []
        for i, row in enumerate(split):
            name = f"{split_name}_{i+1:04d}.png"
            row[key_a].convert("RGB").save(os.path.join(dest_root, split_name, dirs_map[0], name))
            row[key_b].convert("RGB").save(os.path.join(dest_root, split_name, dirs_map[1], name))
            # 标签二值化为 0/255，便于后续 loader 直接除以 255 得 0/1
            mask = row[key_m].convert("L").point(lambda p: 255 if p > 0 else 0)
            mask.save(os.path.join(dest_root, split_name, dirs_map[2], name))
            names.append(name)
        with open(os.path.join(dest_root, f"{split_name}_list.txt"), "w", encoding="utf-8") as f:
            for n in names:
                f.write(n + "\n")
        return True

    ok_train = save_split("train", "train")
    ok_test = save_split("test", "test")
    ok_val = save_split("validation", "val")

    if not ok_val and ok_test:
        for sub in dirs_map:
            src_dir = os.path.join(dest_root, "test", sub)
            dst_dir = os.path.join(dest_root, "val", sub)
            for name in os.listdir(src_dir):
                src = os.path.join(src_dir, name)
                dst = os.path.join(dst_dir, name)
                if not os.path.isfile(dst):
                    from shutil import copy2
                    copy2(src, dst)
        with open(os.path.join(dest_root, "test_list.txt"), "r", encoding="utf-8") as f:
            test_names = [line.strip() for line in f if line.strip()]
        with open(os.path.join(dest_root, "val_list.txt"), "w", encoding="utf-8") as f:
            for n in test_names:
                f.write(n + "\n")

    return dest_root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download LEVIR-CDPlus and organize for ChangeMamba")
    parser.add_argument("--output_dir", type=str, default="~/data",
                        help="Root directory to store the dataset")
    parser.add_argument("--structure", type=str, choices=["T1T2GT", "ABlabel"], default="T1T2GT",
                        help="Folder naming: T1/T2/GT (default) or A/B/label")
    parser.add_argument("--dataset_name", type=str, default="LEVIR-CD+",
                        help="Subfolder name to save the dataset")
    args = parser.parse_args()

    dest = download_levir_cdplus(
        os.path.expanduser(args.output_dir),
        structure=args.structure,
        dataset_name=args.dataset_name
    )
    print(f"Downloaded LEVIR-CD+ to: {dest}")
