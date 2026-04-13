"""Run the DRM simulation."""

import pathlib
import sys

current_path = pathlib.Path(__file__).parent.resolve()
generic_source_path = current_path.parent / "src"
sys.path.append(str(generic_source_path))

from simulation import DRMSimulation

if __name__ == "__main__":
    sim = DRMSimulation(sys.argv[1])
    sim.run()
    sim.postprocess()
    sim.finalise()
