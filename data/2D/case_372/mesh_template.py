# rst Import
import numpy as np
import os
from pyhyp import pyHyp
import prefoil

# rst SurfMesh
coords_fname = "case_coords.dat"
data_path = os.path.join(os.path.dirname(__file__), coords_fname)
coords = prefoil.utils.readCoordFile(data_path)
airfoil = prefoil.Airfoil(coords)
airfoil.makeBluntTE(xCut=0.99)
N_sample = 180
nTEPts = 4

coords = airfoil.getSampledPts(
    N_sample,
    spacingFunc=prefoil.sampling.conical,
    func_args={"coeff": 1.2},
    nTEPts=4,
)

# Write a fitted FFD with 10 chordwise points
ffd_ymarginu = 0.05
ffd_ymarginl = 0.05
ffd_fname = "ffd"
ffd_pts = 10
airfoil.generateFFD(ffd_pts, ffd_fname, ymarginu=ffd_ymarginu, ymarginl=ffd_ymarginl)

# write out plot3d
P3D_fname = "coords.xyz"
P3D_fname_path = os.path.join(os.path.dirname(__file__), P3D_fname)
airfoil.writeCoords(P3D_fname_path[:-4], file_format="plot3d")

N_points = 100
s0 = 4.095080466295228e-06

marchDist = 100.0
# rst GenOptions
options = {
    # ---------------------------
    #        Input Parameters
    # ---------------------------
    "inputFile": P3D_fname_path,
    "unattachedEdgesAreSymmetry": False,
    "outerFaceBC": "farfield",
    "autoConnect": True,
    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
    "families": "wall",
    # rst GridOptions
    # ---------------------------
    #        Grid Parameters
    # ---------------------------
    "N": N_points,
    "nConstantStart": 8,
    "s0": s0,
    "marchDist": marchDist,
    # Smoothing parameters
    "volSmoothIter": 200,
    "volCoef": 0.25,
    "volBlend": 0.001,
    "volSmoothSchedule": [[0, 0], [0.2, 2], [0.5, 200], [1.0, 1000]],
}

# rst Run
hyp = pyHyp(options=options)
hyp.run()
cgns_fname = "case_mesh.cgns"
cgns_fname_path = os.path.join(os.path.dirname(__file__), cgns_fname)
hyp.writeCGNS(cgns_fname_path)