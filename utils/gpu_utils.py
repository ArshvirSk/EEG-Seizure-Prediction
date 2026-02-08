"""
GPU Utilities Module

This module provides GPU configuration and utilities for:
- GPU detection and setup
- Memory management
- Mixed precision training
- Multi-GPU support
"""

import os
import warnings
from typing import List, Optional, Tuple


def setup_gpu(
    memory_growth: bool = True,
    memory_limit: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    mixed_precision: bool = True,
    verbose: bool = True,
) -> Tuple[bool, int]:
    """
    Configure GPU settings for TensorFlow.

    Args:
        memory_growth: Enable memory growth (allocate as needed, not all at once)
        memory_limit: Optional memory limit in MB per GPU (None = no limit)
        gpu_ids: List of GPU IDs to use (None = use all available)
        mixed_precision: Enable mixed precision training (float16 compute, float32 storage)
        verbose: Print GPU configuration info

    Returns:
        Tuple of (gpu_available: bool, num_gpus: int)
    """
    import tensorflow as tf

    # Get list of physical GPUs
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)

    if verbose:
        print("\n" + "=" * 60)
        print("GPU CONFIGURATION")
        print("=" * 60)

    if num_gpus == 0:
        if verbose:
            print("No GPU detected. Running on CPU.")
            print("For GPU support, install: pip install tensorflow[and-cuda]")
        return False, 0

    if verbose:
        print(f"GPUs detected: {num_gpus}")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu.name}")

    try:
        # Filter GPUs if specific IDs requested
        if gpu_ids is not None:
            visible_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]
            tf.config.set_visible_devices(visible_gpus, "GPU")
            gpus = visible_gpus
            if verbose:
                print(f"Using GPUs: {gpu_ids}")

        # Configure each GPU
        for gpu in gpus:
            if memory_growth:
                # Allow memory growth - prevents TF from allocating all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                if verbose:
                    print(f"  Memory growth enabled for {gpu.name}")

            if memory_limit is not None:
                # Set memory limit
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)],
                )
                if verbose:
                    print(f"  Memory limit set to {memory_limit}MB for {gpu.name}")

        # Enable mixed precision training
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            if verbose:
                print("Mixed precision (float16) enabled for faster training")

        if verbose:
            print("=" * 60)

        return True, len(gpus)

    except RuntimeError as e:
        warnings.warn(f"GPU configuration failed: {e}")
        return False, 0


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.

    Returns:
        Dictionary with GPU details
    """
    import tensorflow as tf

    info = {
        "tensorflow_version": tf.__version__,
        "cuda_available": tf.test.is_built_with_cuda(),
        "gpu_available": tf.config.list_physical_devices("GPU"),
        "num_gpus": len(tf.config.list_physical_devices("GPU")),
    }

    # Get GPU memory info if available
    if info["num_gpus"] > 0:
        try:
            from tensorflow.python.client import device_lib

            devices = device_lib.list_local_devices()
            gpu_devices = [d for d in devices if d.device_type == "GPU"]
            info["gpu_details"] = [
                {
                    "name": d.name,
                    "memory_limit": d.memory_limit // (1024 * 1024),  # Convert to MB
                }
                for d in gpu_devices
            ]
        except Exception:
            pass

    return info


def print_gpu_status():
    """Print current GPU status and configuration."""
    info = get_gpu_info()

    print("\n" + "=" * 60)
    print("GPU STATUS")
    print("=" * 60)
    print(f"TensorFlow version: {info['tensorflow_version']}")
    print(f"CUDA built: {info['cuda_available']}")
    print(f"GPUs available: {info['num_gpus']}")

    if info["num_gpus"] > 0 and "gpu_details" in info:
        print("\nGPU Details:")
        for gpu in info["gpu_details"]:
            print(f"  {gpu['name']}: {gpu['memory_limit']}MB")

    print("=" * 60)


def create_distributed_strategy(strategy_type: str = "auto"):
    """
    Create a TensorFlow distribution strategy for multi-GPU training.

    Args:
        strategy_type: 'auto', 'mirrored', 'tpu', or 'default'

    Returns:
        TensorFlow distribution strategy
    """
    import tensorflow as tf

    if strategy_type == "auto":
        # Auto-detect best strategy
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            return tf.distribute.get_strategy()  # Default strategy
        else:
            return tf.distribute.get_strategy()

    elif strategy_type == "mirrored":
        # Multi-GPU synchronous training
        return tf.distribute.MirroredStrategy()

    elif strategy_type == "tpu":
        # TPU training
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver)
        except Exception as e:
            warnings.warn(f"TPU not available: {e}")
            return tf.distribute.get_strategy()

    else:
        # Default single device strategy
        return tf.distribute.get_strategy()


def optimize_for_inference(model, batch_size: int = 1):
    """
    Optimize a trained model for inference.

    Args:
        model: Trained Keras model
        batch_size: Expected inference batch size

    Returns:
        Optimized model
    """
    import tensorflow as tf

    # Convert to TensorFlow Lite for mobile/edge deployment
    # Or use TF-TRT for NVIDIA GPU optimization

    # For now, just compile with optimizations
    try:
        # Enable XLA compilation for faster inference
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    return model


if __name__ == "__main__":
    # Test GPU setup
    print("Testing GPU Utilities...")

    # Check GPU status
    print_gpu_status()

    # Try to setup GPU
    gpu_available, num_gpus = setup_gpu(
        memory_growth=True, mixed_precision=True, verbose=True
    )

    print(f"\nGPU available: {gpu_available}")
    print(f"Number of GPUs: {num_gpus}")

    # Get detailed info
    info = get_gpu_info()
    print(f"\nDetailed info: {info}")

    print("\n✓ GPU utilities test passed!")
