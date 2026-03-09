#!/usr/bin/env python3
"""Create hybrid train/val/test split.

Training: ALL synthetic + real_world 0-1999
Validation: real_world 2000-5999
Test: real_world 6000-7999
"""

import json
from pathlib import Path
from collections import Counter

def load_metadata(split_name):
    """Load metadata from existing split file."""
    metadata_dir = Path("data/metadata")
    with open(metadata_dir / f"{split_name}.json", 'r') as f:
        return json.load(f)

def save_metadata(samples, split_name, output_dir="data/metadata"):
    """Save samples to new metadata file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{split_name}.json"
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_path}")
    return output_path

def main():
    # Load all samples from existing splits
    all_train = load_metadata('train')
    all_val = load_metadata('val')
    all_test = load_metadata('test')
    
    # Combine all samples
    all_samples = all_train + all_val + all_test
    
    print(f"Total samples loaded: {len(all_samples)}")
    
    # Separate by source
    # IMPORTANT: Only use synthetic samples from original train split (those with precomputed tokens)
    synthetic_samples = [s for s in all_train if s['source'] == 'synthetic']
    real_world_samples = [s for s in all_samples if s['source'] == 'real_world']
    
    print(f"Synthetic samples: {len(synthetic_samples)}")
    print(f"Real world samples: {len(real_world_samples)}")
    
    # Sort real_world by file_stem (numeric)
    real_world_samples = sorted(real_world_samples, key=lambda x: int(x['file_stem']))
    
    print("\nFirst 5 real_world samples:")
    for s in real_world_samples[:5]:
        print(f"  {s['id']}: stem={s['file_stem']}")
    
    print("\nLast 5 real_world samples:")
    for s in real_world_samples[-5:]:
        print(f"  {s['id']}: stem={s['file_stem']}")
    
    # Create hybrid splits
    # Training: synthetic from train split + real_world 0-1999 that are in train split
    real_train_stems_set = set([s['file_stem'] for s in all_train if s['source'] == 'real_world'])
    real_train = [s for s in real_world_samples 
                  if int(s['file_stem']) <= 1999 and s['file_stem'] in real_train_stems_set]
    hybrid_train = synthetic_samples + real_train
    
    # Validation: real_world 2000-5999 that are in val split (have tokens)
    real_val_stems_set = set([s['file_stem'] for s in all_val if s['source'] == 'real_world'])
    hybrid_val = [s for s in real_world_samples 
                  if 2000 <= int(s['file_stem']) <= 5999 and s['file_stem'] in real_val_stems_set]
    
    # Test: real_world 6000-7999 that are in test split (have tokens)
    real_test_stems_set = set([s['file_stem'] for s in all_test if s['source'] == 'real_world'])
    hybrid_test = [s for s in real_world_samples 
                   if 6000 <= int(s['file_stem']) <= 7999 and s['file_stem'] in real_test_stems_set]
    
    print("\n" + "="*70)
    print("HYBRID SPLIT STATISTICS")
    print("="*70)
    
    print(f"\nTrain: {len(hybrid_train)} samples")
    print(f"  Synthetic: {len(synthetic_samples)}")
    print(f"  Real world: {len(real_train)}")
    
    print(f"\nVal: {len(hybrid_val)} samples")
    print(f"  Real world: {len(hybrid_val)}")
    
    print(f"\nTest: {len(hybrid_test)} samples")
    print(f"  Real world: {len(hybrid_test)}")
    
    # Save new splits
    print("\n" + "="*70)
    print("SAVING NEW SPLITS")
    print("="*70)
    
    save_metadata(hybrid_train, "hybrid_train")
    save_metadata(hybrid_val, "hybrid_val")
    save_metadata(hybrid_test, "hybrid_test")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print("\nNew metadata files created:")
    print("  - data/metadata/hybrid_train.json")
    print("  - data/metadata/hybrid_val.json")
    print("  - data/metadata/hybrid_test.json")
    print("\nTo use these splits, update your training config:")
    print("  split_train: 'hybrid_train'")
    print("  split_val: 'hybrid_val'")

if __name__ == "__main__":
    main()
