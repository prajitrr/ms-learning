#!/bin/bash
#
# Maximum batch size inference for FULL validation set
#

# Paths
CHECKPOINT="checkpoints/molecular-dit/epoch=14-step=86270-val/loss=0.0495.ckpt"
DATA_PATH="../data/massspecgym_filtered/val.h5"
OUTPUT_DIR="inference_results/val_batch_full"

# Model architecture (must match training config)
HIDDEN_SIZE=512
NUM_HEADS=8
DEPTH=12
MAX_ATOMS=80

# Sampling parameters
NUM_STEPS=100  # Number of ODE integration steps
NUM_SAMPLES_PER_SPECTRUM=10  # Generate 10 structures per spectrum

# GPU optimization
# With 45GB free GPU memory, use large batch size
BATCH_SIZE=64

echo "========================================="
echo "FULL VALIDATION SET INFERENCE"
echo "GPU Memory Available: 45GB"
echo "Using batch size: ${BATCH_SIZE}"
echo "Total spectra: ~45,078"
echo "Total structures to generate: ~450,780"
echo "Estimated time: ~6-12 hours"
echo "========================================="

# Run inference on FULL validation set
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
