import numpy as np
import sys

# ======================================================
# Usage: python inspect_csi.py <path_to_file.npy>
# ======================================================

path = sys.argv[1] if len(sys.argv) > 1 else "csi_data.npy"

data = np.load(path, allow_pickle=True)

print("=" * 60)
print("CSI DATA INSPECTION")
print("=" * 60)

# Handle both array and dict/object types
if isinstance(data, np.ndarray):
    print(f"Type:         numpy array")
    print(f"Shape:        {data.shape}")
    print(f"Dtype:        {data.dtype}")
    print(f"Is complex:   {np.iscomplexobj(data)}")
    print(f"Ndim:         {data.ndim}")
    print()

    print("--- Value Statistics ---")
    if np.iscomplexobj(data):
        amp = np.abs(data)
        print(f"Amplitude min:   {amp.min():.6e}")
        print(f"Amplitude max:   {amp.max():.6e}")
        print(f"Amplitude mean:  {amp.mean():.6e}")
        print(f"Phase min:       {np.angle(data).min():.4f} rad")
        print(f"Phase max:       {np.angle(data).max():.4f} rad")
        # Power in dB
        power = amp**2
        power = np.maximum(power, 1e-30)
        power_db = 10 * np.log10(power)
        print(f"Power (dB) min:  {power_db.min():.2f}")
        print(f"Power (dB) max:  {power_db.max():.2f}")
        print(f"Power (dB) mean: {power_db.mean():.2f}")
    else:
        print(f"Min:    {data.min():.6e}")
        print(f"Max:    {data.max():.6e}")
        print(f"Mean:   {data.mean():.6e}")
        print(f"Std:    {data.std():.6e}")
    print()

    print("--- First Sample Snippet ---")
    # Print first element, navigating up to 3 dims deep
    snippet = data
    for _ in range(min(data.ndim - 1, 3)):
        snippet = snippet[0]
    print(f"data{['[0]']*min(data.ndim-1,3)} shape: {snippet.shape}")
    print(snippet[:5] if snippet.ndim == 1 else snippet[:3, :3])
    print()

    print("--- Dimensions Interpretation ---")
    dims = data.shape
    if data.ndim == 6:
        print(f"Likely: [num_samples, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or similar")
        print(f"  dim0={dims[0]}, dim1={dims[1]}, dim2={dims[2]}, dim3={dims[3]}, dim4={dims[4]}, dim5={dims[5]}")
    elif data.ndim == 5:
        print(f"Likely: [num_samples, num_rx, num_tx, num_subcarriers, num_paths] or similar")
        print(f"  dim0={dims[0]}, dim1={dims[1]}, dim2={dims[2]}, dim3={dims[3]}, dim4={dims[4]}")
    elif data.ndim == 4:
        print(f"Likely: [num_samples, num_rx, num_tx, num_subcarriers/paths]")
        print(f"  dim0={dims[0]}, dim1={dims[1]}, dim2={dims[2]}, dim3={dims[3]}")
    elif data.ndim == 3:
        print(f"Likely: [num_samples, num_rx, feature_dim]")
        print(f"  dim0={dims[0]}, dim1={dims[1]}, dim2={dims[2]}")
    elif data.ndim == 2:
        print(f"Likely: [num_samples, features] or [num_rx, num_tx]")
        print(f"  dim0={dims[0]}, dim1={dims[1]}")

elif isinstance(data, dict) or (hasattr(data, 'item') and isinstance(data.item(), dict)):
    # Handle pickled dict saved as .npy
    d = data.item() if not isinstance(data, dict) else data
    print(f"Type: dict with keys: {list(d.keys())}")
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  '{k}': shape={v.shape}, dtype={v.dtype}, complex={np.iscomplexobj(v)}")
        else:
            print(f"  '{k}': {type(v).__name__} = {v}")

print("=" * 60)