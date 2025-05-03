import json
import cv2
import numpy as np
import os

def load_json_and_image(json_path, image_path):
    """
    Load the JSON file and the corresponding image with error handling.
    """
    try:
        if not json_path.endswith('.json'):
            raise ValueError("The provided file is not a JSON file.")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("✅ JSON loaded successfully.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        print("✅ Image loaded successfully.")

        return data, image

    except Exception as e:
        print(f"❌ Error loading JSON or image: {e}")
        return None, None

def draw_segmentation(data, image):
    """
    Draw segmentation masks on the image based on the JSON shapes and save the output.
    """
    try:
        if data is None or image is None:
            raise ValueError("Data or Image is not properly loaded.")

        mask = np.zeros_like(image)
        print("🛠️ Mask initialized.")

        for shape in data.get("shapes", []):
            label = shape.get("label", "Unknown")
            points = np.array(shape.get("points", []), dtype=np.int32)

            if points.size == 0:
                print(f"⚠️ Skipping empty points for label: {label}")
                continue

            try:
                points = points.reshape((-1, 1, 2))
            except Exception as e:
                print(f"⚠️ Error reshaping points for label {label}: {e}")
                continue

            print(f"✏️ Drawing label: {label}, Points: {points}")
            cv2.fillPoly(mask, [points], (255, 255, 255))

        print("✅ Segmentation drawing completed.")
        output_path = "segmented_output.jpg"
        cv2.imwrite(output_path, mask)
        print(f"📸 Segmented image saved at {output_path}.")

        return mask

    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        return None

def main(json_path, image_path):
    """
    Main function to execute the segmentation pipeline.
    """
    try:
        data, image = load_json_and_image(json_path, image_path)
        if data is None or image is None:
            print("❌ Loading failed. Exiting.")
            return

        print("🚀 Starting segmentation drawing...")
        segmented_image = draw_segmentation(data, image)

        if segmented_image is not None:
            cv2.imshow("Segmented Image", segmented_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Segmentation failed. No output generated.")

    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")

if __name__ == "__main__":
    json_file_path = "path/to/your/json/file.json"  # Replace with actual path
    image_file_path = "path/to/your/image/file.jpg"  # Replace with actual path
    main(json_file_path, image_file_path)
