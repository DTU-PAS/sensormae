"""LLVIP: COCO annotations from the VOC files of the official release.

Expects the official layout with the image folders renamed:

    llvip/Annotations/*.xml
    llvip/{train,test}/rgb/*.jpg     (visible)
    llvip/{train,test}/lwir/*.jpg    (infrared)

and writes llvip/{train,test}/data_coco.json.

    python data_preparation/llvip.py --root data/llvip
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET


def convert(root, split):
    image_names = sorted(os.listdir(os.path.join(root, split, "rgb")))
    images, annotations = [], []
    for image_id, file_name in enumerate(image_names, start=1):
        xml = ET.parse(os.path.join(root, "Annotations", os.path.splitext(file_name)[0] + ".xml")).getroot()
        size = xml.find("size")
        images.append({
            "id": image_id,
            "file_name": file_name,
            "height": int(size.find("height").text),
            "width": int(size.find("width").text),
            "license": None,
            "coco_url": None,
        })
        for obj in xml.iter("object"):
            box = obj.find("bndbox")
            x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })

    coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "person", "supercategory": None}],
        "images": images,
        "annotations": annotations,
    }
    out_path = os.path.join(root, split, "data_coco.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"{out_path}: {len(images)} images, {len(annotations)} boxes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/llvip")
    args = parser.parse_args()
    for split in ("train", "test"):
        convert(args.root, split)


if __name__ == "__main__":
    main()
