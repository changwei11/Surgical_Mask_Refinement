#!/usr/bin/env python3
"""Create symbolic links for inference split tokens."""

from pathlib import Path
import os

def create_inference_token_symlinks():
    token_dir = Path("outputs/clip_tokens")
    inference_dir = token_dir / "inference_real_world_ge2000"
    
    if inference_dir.exists():
        print(f"Token directory {inference_dir} already exists, cleaning up...")
        import shutil
        shutil.rmtree(inference_dir)
    
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    # Create real_world subdirectory that links to all available tokens
    real_world_dir = inference_dir / "real_world"
    real_world_dir.mkdir(exist_ok=True)
    
    # Link all tokens from train/val/test for real_world >= 2000
    linked_count = 0
    for split in ['train', 'val', 'test']:
        split_dir = token_dir / split / "real_world"
        if not split_dir.exists():
            continue
        
        for token_file in split_dir.glob("*.pt"):
            file_stem = int(token_file.stem)
            if file_stem >= 2000:
                dst_path = real_world_dir / token_file.name
                if not dst_path.exists():
                    os.symlink(token_file.absolute(), dst_path)
                    linked_count += 1
    
    print(f"Created {linked_count} token symlinks in {inference_dir}")
    return inference_dir

if __name__ == "__main__":
    inference_dir = create_inference_token_symlinks()
    print(f"\n✓ Inference token directory ready: {inference_dir}")
