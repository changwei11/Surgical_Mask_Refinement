#!/usr/bin/env python3
"""Check which tokens are available for real_world samples >= 2000."""

import json
from pathlib import Path

# Load all metadata
all_samples = []
for split in ['train', 'val', 'test']:
    with open(f'data/metadata/{split}.json') as f:
        all_samples.extend(json.load(f))

# Filter real_world >= 2000
real_ge_2000 = [s for s in all_samples if s['source'] == 'real_world' and int(s['file_stem']) >= 2000]
print(f"Total real_world samples >= 2000: {len(real_ge_2000)}")

# Check token availability
token_counts = {'train': 0, 'val': 0, 'test': 0}
samples_with_tokens = []

for sample in real_ge_2000:
    file_stem = sample['file_stem']
    has_token = False
    
    for split in ['train', 'val', 'test']:
        token_path = Path(f'outputs/clip_tokens/{split}/real_world/{file_stem}.pt')
        if token_path.exists():
            token_counts[split] += 1
            has_token = True
            samples_with_tokens.append(sample)
            break
    
    if not has_token:
        print(f"WARNING: No token found for {sample['id']} (stem={file_stem})")

print(f"\nToken availability by split:")
for split, count in token_counts.items():
    print(f"  {split}: {count} tokens")
print(f"Total with tokens: {sum(token_counts.values())}")
print(f"Missing tokens: {len(real_ge_2000) - len(samples_with_tokens)}")

# Save samples with tokens to inference metadata
if samples_with_tokens:
    output_path = Path('data/metadata/inference_real_world_ge2000.json')
    with open(output_path, 'w') as f:
        json.dump(samples_with_tokens, f, indent=2)
    print(f"\nSaved {len(samples_with_tokens)} samples to {output_path}")
