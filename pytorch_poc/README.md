# Minecraft Texture VAE (PoC)

Proof of concept for a Convolutional Variational Autoencoder (VAE) trained on 16x16 Minecraft item textures.

The goal of this proof of concept is to validate the model architecture before implementing it in C. 

Additionally, this project has been used to extract the minecraft textures from the official source to a more desirable file format.

## Reproduction

To reproduce the results, you will need to download the minecraft vanilla resourcepack. You can find it in this repo https://github.com/Mojang/bedrock-samples.

Place the downloaded zip (e.g. `bedrock-samples-1.21.0.zip`) in this folder. 

This project uses `uv` for python package management.

1.  **Sync environment**:
    ```bash
    uv sync
    ```

2.  **Extract Data**:
    ```bash
    uv run python extract_textures.py
    ```
    This script extracts 16x16 PNGs to `mc_items_png/` and creates a raw binary file `mc_items.bin` (RGB format with a 20-byte header) for easy reading from C.

3.  **Run the PyTorch model**:
    Use the jupyter notebook `vae_poc.ipynb`.
