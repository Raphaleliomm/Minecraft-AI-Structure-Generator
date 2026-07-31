"""Quick test for the create_noisy_blocks function and diffusion training functions."""
import sys
import os
import ast

# Change to the program directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# First, verify syntax of both files
print("Checking syntax...")
ast.parse(open("app/diffusion_model.py", encoding="utf-8").read())
print("  app/diffusion_model.py: OK")
ast.parse(open("kaggle_export.py", encoding="utf-8").read())
print("  kaggle_export.py: OK")

# Now test the create_noisy_blocks function
print("\nTesting create_noisy_blocks...")
import torch
from app.diffusion_model import create_noisy_blocks

# Create a small test: batch of 10 samples, 4x4x4 grid, 5 block types
B = 10
GX, GY, GZ = 4, 4, 4
num_blocks = 5
num_timesteps = 50

# Clean target: all blocks are type 1 (stone)
clean = torch.ones((B, GX, GY, GZ), dtype=torch.long)

# Timesteps: half at t=10, half at t=40
timesteps = torch.cat([torch.full((5,), 10), torch.full((5,), 40)])

# Test with noise_block_prob=0.20 (20% of samples get random blocks)
noisy = create_noisy_blocks(clean, timesteps, num_timesteps, num_blocks, noise_block_prob=0.20)

print(f"  Input shape: {noisy.shape}")
print(f"  Clean values: {set(clean.flatten().tolist())}")

# At t=10, noise fraction should be ~10/50 = 0.2
# At t=40, noise fraction should be ~40/50 = 0.8
# Some samples should have air (0), some should have random blocks (1-4)

# Check that noisy has values in valid range
assert noisy.min() >= 0, f"Min value {noisy.min()} < 0"
assert noisy.max() < num_blocks, f"Max value {noisy.max()} >= {num_blocks}"
print(f"  Noisy value range: [{noisy.min()}, {noisy.max()}]")

# Test with noise_block_prob=0.0 (no block injection, standard air noise)
noisy_standard = create_noisy_blocks(clean, timesteps, num_timesteps, num_blocks, noise_block_prob=0.0)
# All noise positions should be air (0)
noise_positions = noisy_standard != clean
noise_values = noisy_standard[noise_positions]
if len(noise_values) > 0:
    assert noise_values.min() == 0 and noise_values.max() == 0, "Standard noise should only have air (0)"
    print(f"  Standard noise (prob=0.0): all noise positions are air (0) OK")

# Test with noise_block_prob=1.0 (all samples get random blocks)
noisy_all = create_noisy_blocks(clean, timesteps, num_timesteps, num_blocks, noise_block_prob=1.0)
noise_positions = noisy_all != clean
noise_values = noisy_all[noise_positions]
if len(noise_values) > 0:
    # Should have non-air blocks (1 to num_blocks-1)
    assert noise_values.min() >= 1, f"Injected blocks should be >= 1, got min={noise_values.min()}"
    print(f"  Block injection (prob=1.0): all noise positions have random blocks [{noise_values.min()}, {noise_values.max()}] OK")

# Test edge case: t=0 (no noise)
noisy_t0 = create_noisy_blocks(clean, torch.zeros(B, dtype=torch.long), num_timesteps, num_blocks, noise_block_prob=0.20)
assert torch.equal(noisy_t0, clean), "At t=0, no noise should be added"
print(f"  Edge case t=0: no noise added OK")

# Test edge case: t=num_timesteps-1 (maximum noise)
noisy_tmax = create_noisy_blocks(clean, torch.full((B,), num_timesteps - 1, dtype=torch.long), num_timesteps, num_blocks, noise_block_prob=0.20)
noise_fraction = (noisy_tmax != clean).float().mean().item()
print(f"  Edge case t=max: noise fraction = {noise_fraction:.2f} (should be close to 1.0)")

print("\nAll tests passed!")