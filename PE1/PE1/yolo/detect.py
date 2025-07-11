from ultralytics import YOLO
import os
from PIL import Image, ImageDraw

def label_and_save(image_path, boxes, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

    image.save(save_path)

model = YOLO('yolo/tree-detection-model.pt') 

input_dir = 'static/images/uploads'
output_dir = 'static/images/results'
os.makedirs(output_dir, exist_ok=True)

def process_images(input_dir, output_dir):
    tree_counts = {}

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        if os.path.isfile(image_path):
            total_trees = 0

            results = model.predict(source=image_path, conf=0.02, iou=0.01, max_det=1000)

            total_trees = sum(len(result.boxes) for result in results)

            save_path = os.path.join(output_dir, image_name)
            for result in results:
                label_and_save(image_path, result.boxes, save_path)

            tree_counts[image_name] = total_trees
            with open(os.path.join(output_dir, f"{image_name}_tree_count.txt"), "w") as f:
                f.write(str(total_trees))
            
            print(f"Processed {image_name}: {total_trees} trees detected.")

    return tree_counts

if __name__ == "__main__":
    results = process_images(input_dir, output_dir)
    print("All images processed.")
    print("Tree counts:", results)