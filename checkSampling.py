import h5py
import sys

f = h5py.File("./datasets/dataset_1_77.hdf5", "r")
data = f["data"]

for demo_key in sorted(data.keys()):
    demo = data[demo_key]
    num_steps = demo["actions"].shape[0]
    
    # Check if timestamps exist
    if "timestamps" in demo:
        ts = demo["timestamps"][:]
        duration = ts[-1] - ts[0]
        print(f"{demo_key}: {num_steps} steps, duration={duration:.2f}s sim time")
    else:
        sim_time = num_steps * 0.05  # decimation=5, dt=1/100
        print(f"{demo_key}: {num_steps} steps, sim_time={sim_time:.2f}s (computed)")
    
    # Check what obs keys exist
    if "obs" in demo:
        print(f"  obs keys: {list(demo['obs'].keys())}")
        for k in demo["obs"].keys():
            print(f"    {k}: shape={demo['obs'][k].shape}")

f.close()