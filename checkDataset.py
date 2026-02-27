import h5py

with h5py.File("output_dataset.hdf5", "r") as f:
    frames = f["data/demo_0/obs/cam_egoview_rgb"]
    print(frames.shape)



with h5py.File("output_dataset.hdf5", "r") as f:
    def print_tree(name):
        print(name)
    f.visit(print_tree)