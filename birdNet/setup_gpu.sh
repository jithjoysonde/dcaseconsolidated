#!/bin/bash
# GPU Setup Script for BirdNet Processing
# This script verifies GPU is available for TensorFlow

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source dcase_t5/bin/activate
fi

# Suppress TensorFlow info messages
export TF_CPP_MIN_LOG_LEVEL=1

echo "Checking GPU availability..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'✓ GPU ready: {len(gpus)} device(s) found')
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        print(f'  GPU {i}: {details.get(\"device_name\", \"Unknown\")} - Compute Capability {details.get(\"compute_capability\", \"N/A\")}')
else:
    print('✗ No GPU found - will use CPU only')
"

echo ""
echo "GPU setup complete!"
