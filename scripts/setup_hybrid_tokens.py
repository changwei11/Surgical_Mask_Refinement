#!/usr/bin/env python3
"""Create symbolic links for hybrid split tokens.

Since tokens are already precomputed in train/val/test splits,
we create symlinks so hybrid splits can access them.
"""

from pathlib import Path
import os

def main():
    token_dir = Path("outputs/clip_tokens")
    
    if not token_dir.exists():
        print(f"ERROR: Token directory not found: {token_dir}")
        print("Please run token precomputation first.")
        return
    
    print("="*70)
    print("CREATING TOKEN SYMLINKS FOR HYBRID SPLITS")
    print("="*70)
    print()
    
    # hybrid_train: uses tokens from train split (both synthetic and real_world)
    hybrid_train_dir = token_dir / "hybrid_train"
    if not hybrid_train_dir.exists():
        print(f"Creating {hybrid_train_dir}/")
        hybrid_train_dir.mkdir(parents=True, exist_ok=True)
        
        # Symlink synthetic
        synthetic_src = token_dir / "train" / "synthetic"
        synthetic_dst = hybrid_train_dir / "synthetic"
        if synthetic_src.exists() and not synthetic_dst.exists():
            os.symlink(synthetic_src.absolute(), synthetic_dst)
            print(f"  ✓ Linked {synthetic_dst} -> {synthetic_src}")
        
        # Symlink real_world
        real_src = token_dir / "train" / "real_world"
        real_dst = hybrid_train_dir / "real_world"
        if real_src.exists() and not real_dst.exists():
            os.symlink(real_src.absolute(), real_dst)
            print(f"  ✓ Linked {real_dst} -> {real_src}")
    else:
        print(f"  ✓ {hybrid_train_dir}/ already exists")
    
    print()
    
    # hybrid_val: uses tokens from val split (real_world only)
    hybrid_val_dir = token_dir / "hybrid_val"
    if not hybrid_val_dir.exists():
        print(f"Creating {hybrid_val_dir}/")
        hybrid_val_dir.mkdir(parents=True, exist_ok=True)
        
        # Symlink real_world
        real_src = token_dir / "val" / "real_world"
        real_dst = hybrid_val_dir / "real_world"
        if real_src.exists() and not real_dst.exists():
            os.symlink(real_src.absolute(), real_dst)
            print(f"  ✓ Linked {real_dst} -> {real_src}")
    else:
        print(f"  ✓ {hybrid_val_dir}/ already exists")
    
    print()
    
    # hybrid_test: uses tokens from test split (real_world only)
    hybrid_test_dir = token_dir / "hybrid_test"
    if not hybrid_test_dir.exists():
        print(f"Creating {hybrid_test_dir}/")
        hybrid_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Symlink real_world
        real_src = token_dir / "test" / "real_world"
        real_dst = hybrid_test_dir / "real_world"
        if real_src.exists() and not real_dst.exists():
            os.symlink(real_src.absolute(), real_dst)
            print(f"  ✓ Linked {real_dst} -> {real_src}")
    else:
        print(f"  ✓ {hybrid_test_dir}/ already exists")
    
    print()
    print("="*70)
    print("SUCCESS!")
    print("="*70)
    print()
    print("Token directory structure:")
    print(f"  {token_dir}/")
    print(f"    hybrid_train/")
    print(f"      synthetic/ -> train/synthetic/")
    print(f"      real_world/ -> train/real_world/")
    print(f"    hybrid_val/")
    print(f"      real_world/ -> val/real_world/")
    print(f"    hybrid_test/")
    print(f"      real_world/ -> test/real_world/")
    print()
    print("Ready to train with hybrid splits!")

if __name__ == "__main__":
    main()
