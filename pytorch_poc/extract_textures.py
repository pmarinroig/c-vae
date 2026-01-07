import zipfile
import os
import io
import glob
import struct
from PIL import Image
import shutil

def extract_mc_textures(zip_pattern="bedrock-samples*.zip", output_dir="mc_items_png", bin_output="mc_items.bin", manifest_output="mc_items.txt"):
    # Find zip file matching pattern
    zip_files = glob.glob(zip_pattern)
    if not zip_files:
        # Try parent directory
        zip_files = glob.glob(os.path.join("..", zip_pattern))
    
    if not zip_files:
        print(f"No zip file matching '{zip_pattern}' found.")
        return

    # Pick the first match
    zip_path = zip_files[0]
    print(f"Using resource pack: {zip_path}")

    # Clean output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Cleaned and created directory '{output_dir}'.")

    # List to store (filename, bytes) tuples
    valid_images = []
    
    print(f"Scanning '{zip_path}' for 16x16 item textures (root folder only)...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the main textures/item folder
        target_dir = None
        for name in z.namelist():
            norm_name = name.replace('\\', '/')
            if norm_name.endswith('textures/item/') or norm_name.endswith('textures/items/'):
                target_dir = norm_name
                print(f"Found item texture directory: {target_dir}")
                break
        
        if not target_dir:
            print("Could not find a 'textures/item/' folder in the zip.")
            return

        expected_dir = target_dir.rstrip('/')

        for file_info in z.infolist():
            if file_info.is_dir():
                continue
                
            norm_name = file_info.filename.replace('\\', '/')
            file_dir = os.path.dirname(norm_name)
            
            if file_dir == expected_dir and norm_name.endswith('.png'):
                try:
                    with z.open(file_info) as file:
                        img_data = file.read()
                        with Image.open(io.BytesIO(img_data)) as img:
                            if img.size == (16, 16):
                                # Convert to RGB for consistency (handles transparency by making it black)
                                rgb_img = img.convert('RGB')
                                
                                # Save PNG
                                base_name = os.path.basename(file_info.filename)
                                target_path = os.path.join(output_dir, base_name)
                                with open(target_path, "wb") as f:
                                    # We re-save the RGB version to ensure consistency
                                    rgb_img.save(f, format="PNG")
                                
                                # Store filename and raw bytes
                                valid_images.append((base_name, rgb_img.tobytes()))
                                
                except Exception as e:
                    print(f"Skipping {file_info.filename}: {e}")

    count = len(valid_images)
    print(f"Extraction complete. Found {count} valid 16x16 textures.")

    # Create C-friendly binary file
    # Header: Magic(4s), Count(I), Width(I), Height(I), Channels(I)
    print(f"Creating binary dataset '{bin_output}'...")
    with open(bin_output, "wb") as f:
        # Magic "MCVA"
        f.write(b'MCVA')
        # Count
        f.write(struct.pack('<I', count))
        # Width, Height, Channels
        f.write(struct.pack('<III', 16, 16, 3))
        
        # Data
        for _, img_bytes in valid_images:
            f.write(img_bytes)
            
    print(f"Binary dataset saved. Size: {os.path.getsize(bin_output)} bytes.")

    # Create Manifest file
    print(f"Creating manifest '{manifest_output}'...")
    with open(manifest_output, "w") as f:
        for name, _ in valid_images:
            f.write(f"{name}\n")
    print("Manifest saved.")

if __name__ == "__main__":
    extract_mc_textures()