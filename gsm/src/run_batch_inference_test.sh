#!/bin/bash
#
# Maximum batch size inference for testing (1000 spectra)
#

# Paths
CHECKPOINT="checkpoints/molecular-dit/epoch=14-step=86270-val/loss=0.0495.ckpt"
DATA_PATH="../data/massspecgym_filtered/val.h5"
OUTPUT_DIR="inference_results/val_batch_test"

# Model architecture (must match training config)
HIDDEN_SIZE=512
NUM_HEADS=8
DEPTH=12
MAX_ATOMS=80

# Sampling parameters
NUM_STEPS=100  # Number of ODE integration steps
NUM_SAMPLES_PER_SPECTRUM=10  # Generate 10 structures per spectrum

# GPU optimization
# With 45GB free, we can handle much larger batches
# Model forward pass per sample: ~500MB for batch_size=8
# Let's try batch_size=64 (8x larger) = ~4GB, leaving room for safety
BATCH_SIZE=64

echo "========================================="
echo "GPU Memory Available: 45GB"
echo "Using batch size: ${BATCH_SIZE}"
echo "========================================="

# Run inference on subset (1000 samples for testing)
python inference.py \
    --checkpoint ${CHECKPOINT} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_heads ${NUM_HEADS} \
    --depth ${DEPTH} \
    --max_atoms ${MAX_ATOMS} \
    --num_steps ${NUM_STEPS} \
    --num_samples_per_spectrum ${NUM_SAMPLES_PER_SPECTRUM} \
    --num_samples 1000 \
    --batch_size ${BATCH_SIZE} \
    --refine \
    --guess_bonds \
    --force_field MMFF \
    --refine_iters 1000 \
    --seed 42

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}"
echo ""
echo "Run analysis with:"
echo "  python analyze_topk.py ${OUTPUT_DIR}"
