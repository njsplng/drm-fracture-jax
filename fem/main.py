"""Run the FEM simulation."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pathlib
import sys

current_path = pathlib.Path(__file__).parent.resolve()
generic_source_path = current_path.parent / "src"
sys.path.append(str(generic_source_path))

from simulation import FEMSimulation

if __name__ == "__main__":
    sim = FEMSimulation(sys.argv[1])
    sim.run()
    sim.postprocess()
    sim.finalise()
