import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

def create_visualization(base_name, minus_name, plus_name, output_file="visualization.png"):
    files = ["base.ppm", "minus.ppm", "plus.ppm", "result.ppm"]
    titles = [f"Base:\n{base_name}", f"Minus:\n{minus_name}", f"Plus:\n{plus_name}", "Result"]
    
    # Check in current or parent directory
    paths = []
    for f in files:
        if os.path.exists(f):
            paths.append(f)
        elif os.path.exists(os.path.join("..", f)):
            paths.append(os.path.join("..", f))
        else:
            print(f"Error: Missing {f}")
            return

    # Increase width to make space for signs
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.6)

    for i, ax in enumerate(axes):
        img = Image.open(paths[i])
        # Scale up for visibility (16x16 is tiny)
        img = img.resize((128, 128), resample=Image.NEAREST)
        ax.imshow(img)
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')
    
    # Add arithmetic signs centered between plots
    # Axes coordinates are 0 to 1 within the figure width
    # 4 plots. Centers roughly at 0.125, 0.375, 0.625, 0.875
    # Gaps at 0.25, 0.5, 0.75
    
    # The wspace adjustment might shift them, so standard figtext is safer if we eye-ball it or calculate.
    # Let's try 0.3, 0.52, 0.74 roughly. 
    
    plt.figtext(0.31, 0.5, "-", fontsize=40, ha='center', va='center', weight='bold')
    plt.figtext(0.51, 0.5, "+", fontsize=40, ha='center', va='center', weight='bold')
    plt.figtext(0.72, 0.5, "=", fontsize=40, ha='center', va='center', weight='bold')

    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python visualize_results.py <base_name> <minus_name> <plus_name>")
    else:
        create_visualization(sys.argv[1], sys.argv[2], sys.argv[3])