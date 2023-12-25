import numpy as np
import h5py
import pytenet as ptn
from util import interleave_complex


def mps_to_statevector_data():

    # random number generator
    rng = np.random.default_rng(531)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD = [rng.integers(-1, 2, size=Di) for Di in (1, 7, 10, 11, 5, 1)]

    # create a random matrix product state
    mps = ptn.MPS(qd, qD, fill="random", rng=rng)

    # convert to a state vector
    vec = mps.as_vector()

    with h5py.File("data/test_mps_to_statevector.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.A):
            file[f"a{i}"] = interleave_complex(ai)
        file["vec"] = interleave_complex(vec)


def main():
    mps_to_statevector_data()


if __name__ == "__main__":
    main()