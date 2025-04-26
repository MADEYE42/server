import json
import cv2
import numpy as np
import os

def load_json_and_image(json_path, image_path):
    """
    Load the JSON file and the corresponding image with error handling.
    """
    try:
        # Validate file paths
        if not json_path.endswith('.json'):
            raise ValueError("The provided JSON file is not a valid JSON file.")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("JSON loaded successfully.")

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image file: {image_path}")
        print("Image loaded successfully.")

        return data, image

    except Exception as e:
        print(f"Error loading JSON or image: {e}")
        return None, None

def draw_segmentation(data, image):
    """
    Draw segmentation masks on the image based on the JSON shapes and save the output.
    """
    try:
        if data is None or image is None:
            raise ValueError("Data or Image is not properly loaded.")

        # Create an empty mask with the same shape as the image
        mask = np.zeros_like(image)
        print("Mask initialized.")

        for shape in data.get("shapes", []):
            label = shape.get("label", "Unknown")
            points = np.array(shape.get("points", []), dtype=np.int32)

            # Validate points
            if points.size == 0:
                print(f"Skipping empty points for label: {label}")
                continue

            # Ensure proper reshaping
            try:
                points = points.reshape((-1, 1, 2))
            except Exception as e:
                print(f"Error reshaping points for label {label}: {e}")
                continue

            print(f"Drawing label: {label}, Points: {points}")

            # Draw polygon on the mask
            cv2.fillPoly(mask, [points], (255, 255, 255))

        print("Segmentation drawing completed.")

        # Save the segmented image
        output_path = "segmented_output.jpg"
        cv2.imwrite(output_path, mask)
        print(f"Segmented image saved at {output_path}.")
        
        return mask

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None

def main(json_path, image_path):
    """
    Main function to execute the pipeline.
    """
    try:
        # Load JSON and Image
        data, image = load_json_and_image(json_path, image_path)
        if data is None or image is None:
            print("Loading failed. Exiting.")
            return

        # Draw segmentation
        print("Starting segmentation drawing...")
        segmented_image = draw_segmentation(data, image)

        if segmented_image is not None:
            # Display the segmented image
            cv2.imshow("Segmented Image", segmented_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Segmentation failed. No output generated.")

    except Exception as e:
        print(f"Error in main pipeline: {e}")

if __name__ == "__main__":
    # Provide the paths to your JSON and Image files
    json_file_path = "path/to/your/json/file.json"  # Replace with actual path
    image_file_path = "path/to/your/image/file.jpg"  # Replace with actual path
    main(json_file_path, image_file_path)
