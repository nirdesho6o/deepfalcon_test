import h5py

file_path = "data/quark-gluon.hdf5"

with h5py.File(file_path, "r") as f:

    print("Top level keys:")
    print(list(f.keys()))

    for key in f.keys():
        print("\nDataset:", key)
        print("Shape:", f[key].shape)