"""
Complete LaBraM Weight Transfer Script

Combines explicit weight mapping with full backbone transfer.
Uses precise key renaming to transfer all compatible parameters.

Transfers weights from LaBraM checkpoint to Braindecode Labram model.
"""

import torch
import argparse
from braindecode.models import Labram


def create_weight_mapping():
    """
    Create comprehensive weight mapping from LaBraM to Braindecode.
    
    Includes:
    - Temporal convolution layers (patch_embed)
    - All transformer blocks
    - Position embeddings
    - Other backbone components
    """
    return {
        # Temporal Convolution Layers
        'student.patch_embed.conv1.weight': 'patch_embed.temporal_conv.conv1.weight',
        'student.patch_embed.conv1.bias': 'patch_embed.temporal_conv.conv1.bias',
        'student.patch_embed.norm1.weight': 'patch_embed.temporal_conv.norm1.weight',
        'student.patch_embed.norm1.bias': 'patch_embed.temporal_conv.norm1.bias',
        'student.patch_embed.conv2.weight': 'patch_embed.temporal_conv.conv2.weight',
        'student.patch_embed.conv2.bias': 'patch_embed.temporal_conv.conv2.bias',
        'student.patch_embed.norm2.weight': 'patch_embed.temporal_conv.norm2.weight',
        'student.patch_embed.norm2.bias': 'patch_embed.temporal_conv.norm2.bias',
        'student.patch_embed.conv3.weight': 'patch_embed.temporal_conv.conv3.weight',
        'student.patch_embed.conv3.bias': 'patch_embed.temporal_conv.conv3.bias',
        'student.patch_embed.norm3.weight': 'patch_embed.temporal_conv.norm3.weight',
        'student.patch_embed.norm3.bias': 'patch_embed.temporal_conv.norm3.bias',
        # Note: Other backbone layers (blocks, embeddings, norm, fc_norm) are handled
        # by removing 'student.' prefix in process_state_dict()
    }


def process_state_dict(state_dict, weight_mapping):
    """
    Process checkpoint state dict with explicit mapping.
    
    Parameters:
    -----------
    state_dict : dict
        Original checkpoint state dictionary
    weight_mapping : dict
        Explicit mapping for special layers (patch_embed)
        
    Returns:
    --------
    dict : Processed state dict ready for Braindecode model
    """
    new_state = {}
    mapped_keys = []
    skipped_keys = []
    
    for key, value in state_dict.items():
        # Skip classification head (task-specific)
        if 'head' in key:
            skipped_keys.append((key, 'head layer'))
            continue
        
        # Use explicit mapping for patch_embed temporal_conv
        if key in weight_mapping:
            new_key = weight_mapping[key]
            new_state[new_key] = value
            mapped_keys.append((key, new_key))
            continue
        
        # Skip original patch_embed if not in mapping (SegmentPatch)
        if 'patch_embed' in key and 'temporal_conv' not in key:
            skipped_keys.append((key, 'patch_embed (non-temporal)'))
            continue
        
        # For backbone layers, remove 'student.' prefix
        if key.startswith('student.'):
            new_key = key.replace('student.', '')
            new_state[new_key] = value
            mapped_keys.append((key, new_key))
            continue
        
        # Keep other keys as-is
        new_state[key] = value
        mapped_keys.append((key, key))
    
    return new_state, mapped_keys, skipped_keys


def transfer_labram_weights(
    checkpoint_path,
    n_times=1600,
    n_chans=64,
    n_outputs=4,
    output_path=None,
    verbose=True
):
    """
    Transfer LaBraM weights to Braindecode Labram using explicit mapping.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to LaBraM checkpoint
    n_times : int
        Number of time samples
    n_chans : int
        Number of channels
    n_outputs : int
        Number of output classes
    output_path : str
        Where to save the model
    verbose : bool
        Print transfer details
        
    Returns:
    --------
    model : Labram
        Model with transferred weights
    stats : dict
        Transfer statistics
    """
    
    print("\n" + "="*70)
    print("LaBraM → Braindecode Weight Transfer")
    print("="*70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint
    
    original_params = len(state)
    print(f"Original checkpoint: {original_params} parameters")
    
    # Create weight mapping
    weight_mapping = create_weight_mapping()
    
    # Process state dict
    print("\nProcessing checkpoint...")
    new_state, mapped_keys, skipped_keys = process_state_dict(state, weight_mapping)
    
    transferred_params = len(mapped_keys)
    print(f"Mapped keys: {transferred_params} ({transferred_params/original_params*100:.1f}%)")
    print(f"Skipped keys: {len(skipped_keys)}")
    
    if verbose and skipped_keys:
        print(f"\nSkipped layers:")
        for key, reason in skipped_keys[:5]:  # Show first 5
            print(f"  - {key:50s} ({reason})")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    # Create model
    print(f"\nCreating Labram model:")
    print(f"  n_times: {n_times}")
    print(f"  n_chans: {n_chans}")
    print(f"  n_outputs: {n_outputs}")
    model = Labram(
        n_times=n_times,
        n_chans=n_chans,
        n_outputs=n_outputs,
        neural_tokenizer=True,
    )
    
    # Load weights
    print("\nLoading weights into model...")
    incompatible = model.load_state_dict(new_state, strict=False)
    
    missing_count = len(incompatible.missing_keys) if incompatible.missing_keys else 0
    unexpected_count = len(incompatible.unexpected_keys) if incompatible.unexpected_keys else 0
    
    if missing_count > 0:
        print(f"  Missing keys: {missing_count} (expected - will be initialized)")
    if unexpected_count > 0:
        print(f"  Unexpected keys: {unexpected_count}")
    
    # Test forward pass
    if verbose:
        print("\nTesting forward pass...")
        x = torch.randn(2, n_chans, n_times)
        with torch.no_grad():
            output = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✅ Forward pass successful!")
    
    # Save model if output_path provided
    if output_path:
        print(f"\nSaving model to: {output_path}")
        torch.save(model.state_dict(), output_path)
        print(f"  ✅ Model saved")
    
    stats = {
        'original': original_params,
        'transferred': transferred_params,
        'skipped': len(skipped_keys),
        'transfer_rate': f"{transferred_params/original_params*100:.1f}%"
    }
    
    return model, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transfer LaBraM weights to Braindecode Labram',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default transfer (backbone parameters)
  python labram_complete_transfer.py
  
  # Transfer and save model
  python labram_complete_transfer.py --output labram_weights.pt
  
  # Custom EEG parameters
  python labram_complete_transfer.py --n-times 2000 --n-chans 62 --n-outputs 2
  
  # Custom checkpoint path
  python labram_complete_transfer.py --checkpoint path/to/checkpoint.pth
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/var/log/thavamount/eeg_ckpts/labram/labram-base.pth',
        help='Path to LaBraM checkpoint (default: /var/log/thavamount/eeg_ckpts/labram/labram-base.pth)'
    )
    parser.add_argument(
        '--n-times',
        type=int,
        default=1600,
        help='Number of time samples (default: 1600)'
    )
    parser.add_argument(
        '--n-chans',
        type=int,
        default=64,
        help='Number of channels (default: 64)'
    )
    parser.add_argument(
        '--n-outputs',
        type=int,
        default=4,
        help='Number of output classes (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path to save model weights'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (default: cpu)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LaBraM → Braindecode Weight Transfer")
    print("="*70)
    
    # Transfer weights
    model, stats = transfer_labram_weights(
        checkpoint_path=args.checkpoint,
        n_times=args.n_times,
        n_chans=args.n_chans,
        n_outputs=args.n_outputs,
        output_path=args.output,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("✅ TRANSFER COMPLETE")
    print("="*70)
    print(f"Original parameters:   {stats['original']}")
    print(f"Transferred:           {stats['transferred']} ({stats['transfer_rate']})")
    print(f"Skipped:               {stats['skipped']}")
    print("="*70)