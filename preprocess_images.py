import os
import cv2
import glob

def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    """
    Preprocess images from input_dir and save to output_dir.
    Applies grayscale, Otsu's thresholding, and resizing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use glob to find common image formats, ignoring non-images like Thumbs.db
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's Thresholding (to get clean black signature on white background)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Resize
        resized = cv2.resize(thresh, img_size)
        
        # Save output
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, resized)
        
    print(f"Processed {len(image_paths)} images from {input_dir} to {output_dir}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_forg_dir = os.path.join(base_dir, 'signatures', 'full_forg')
    raw_org_dir = os.path.join(base_dir, 'signatures', 'full_org')
    
    proc_forg_dir = os.path.join(base_dir, 'processed_data', 'full_forg')
    proc_org_dir = os.path.join(base_dir, 'processed_data', 'full_org')
    
    print("Starting image preprocessing...")
    preprocess_images(raw_org_dir, proc_org_dir)
    preprocess_images(raw_forg_dir, proc_forg_dir)
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()
