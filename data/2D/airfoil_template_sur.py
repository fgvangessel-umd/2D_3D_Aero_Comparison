# ======================================================================
#         Import modules
# ======================================================================
# rst Imports (beg)
import os
import numpy as np
import argparse
import ast
from mpi4py import MPI
from baseclasses import AeroProblem
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints
from pyoptsparse import Optimization, OPT
from idwarp import USMesh
from multipoint import multiPointSparse

# default_same_dir = True
mesh_fname = "case_mesh.cgns"
FFD_fname = "ffd.xyz"
# Use Python's built-in Argument parser to get commandline options
parser = argparse.ArgumentParser()
# paths to various files
output_path = os.path.join(os.path.dirname(__file__))
mesh_input_path = os.path.join(os.path.dirname(__file__), mesh_fname)
FFD_path = os.path.join(os.path.dirname(__file__), FFD_fname)
args_opt = "SLSQP"

parser.add_argument("--output", type=str, default=output_path)
parser.add_argument("--opt", type=str, default=args_opt, choices=["SLSQP", "PSQP", "IPOPT", "SNOPT", "PAROPT"])
parser.add_argument("--gridFile", type=str, default=mesh_input_path)
parser.add_argument("--optOptions", type=ast.literal_eval, default={}, help="additional optimizer options to be added")
args = parser.parse_args()
# rst args (end)

# ======================================================================
#         Specify parameters for optimization
# ======================================================================
# rst parameters (beg)
# cL constraint
mycl = 0.7875943806850612
# mach number
mach = 0.5419459139620574
# Reynolds number
reynolds = 6588881.514112889
# Reynold's Length
reynoldsLength = 1.0
# Temperature
T = 300.0
# Volume Ratio
v = 0.8531104311138716
# rst parameters (end)
# ======================================================================
#         Create multipoint communication object
# ======================================================================
# rst multipoint (beg)
MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
if not os.path.exists(args.output):
    if comm.rank == 0:
        os.mkdir(args.output)

# rst multipoint (end)
# ======================================================================
#         ADflow Set-up
# ======================================================================
# rst adflow (beg)
aeroOptions = {
    # Common Parameters
    "gridFile": args.gridFile,
    "outputDirectory": args.output,
    "writeSurfaceSolution": False,
    "writeVolumeSolution": False,
    "writeTecplotSurfaceSolution": True,
    "monitorvariables": ["cl", "cd", "yplus"],
    # Physics Parameters
    "equationType": "RANS",
    "smoother": "DADI",
    "nCycles": 5000,
    "rkreset": True,
    "nrkreset": 10,
    # NK Options
    "useNKSolver": True,
    "nkswitchtol": 1e-8,
    # ANK Options
    "useanksolver": True,
    "ankswitchtol": 1e-1,
    # "ANKCoupledSwitchTol": 1e-6,
    # "ANKSecondOrdSwitchTol": 1e-5,
    "liftIndex": 2,
    "infchangecorrection": True,
    "nsubiterturb": 10,
    # Convergence Parameters
    "L2Convergence": 1e-8,
    "L2ConvergenceCoarse": 1e-4,
    # Adjoint Parameters
    "adjointSolver": "GMRES",
    "adjointL2Convergence": 1e-8,
    "ADPC": True,
    "adjointMaxIter": 1000,
    "adjointSubspaceSize": 200,
}

# Create solver
CFDSolver = ADFLOW(options=aeroOptions, comm=comm)
# rst adflow (end)
# ======================================================================
#         Set up flow conditions with AeroProblem
# ======================================================================
# rst aeroproblem (beg)
ap = AeroProblem(
    name="fc",
    alpha=2.5,
    mach=mach,
    T=T,
    reynolds=reynolds,
    reynoldsLength=reynoldsLength,
    areaRef=1.0,
    chordRef=1.0,
    evalFuncs=["cl", "cd"],
)
# Add angle of attack variable
ap.addDV("alpha", value=2.5, lower=0.0, upper=10, scale=1.0)
# rst aeroproblem (end)
# ======================================================================
#         Geometric Design Variable Set-up
# ======================================================================
# rst dvgeo (beg)
# Create DVGeometry object
DVGeo = DVGeometry(FFD_path)
DVGeo.addLocalDV("shape", lower=-0.025, upper=0.025, axis="y", scale=1.0)

span = 1.0
pos = np.array([0.5]) * span
CFDSolver.addSlices("z", pos, sliceType="absolute")

# Add DVGeo object to CFD solver
CFDSolver.setDVGeo(DVGeo)
# rst dvgeo (end)
# ======================================================================
#         DVConstraint Setup
# ======================================================================
# rst dvcon (beg)

DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)

# Only ADflow has the getTriangulatedSurface Function
DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())

# Le/Te constraints
lIndex = DVGeo.getLocalIndex(0)
indSetA = []
indSetB = []
for k in range(0, 1):
    indSetA.append(lIndex[0, 0, k])  # all DV for upper and lower should be same but different sign
    indSetB.append(lIndex[0, 1, k])
for k in range(0, 1):
    indSetA.append(lIndex[-1, 0, k])
    indSetB.append(lIndex[-1, 1, k])
DVCon.addLeTeConstraints(0, indSetA=indSetA, indSetB=indSetB)

# DV should be same along spanwise
lIndex = DVGeo.getLocalIndex(0)
indSetA = []
indSetB = []
for i in range(lIndex.shape[0]):
    indSetA.append(lIndex[i, 0, 0])
    indSetB.append(lIndex[i, 0, 1])
for i in range(lIndex.shape[0]):
    indSetA.append(lIndex[i, 1, 0])
    indSetB.append(lIndex[i, 1, 1])
DVCon.addLinearConstraintsShape(indSetA, indSetB, factorA=1.0, factorB=-1.0, lower=0, upper=0)

le = 0.010001
leList = [[le, 0, le], [le, 0, 1.0 - le]]
teList = [[1.0 - le, 0, le], [1.0 - le, 0, 1.0 - le]]

DVCon.addVolumeConstraint(leList, teList, 2, 100, lower=v, upper=1.20, scaled=True)
DVCon.addThicknessConstraints2D(leList, teList, 2, 100, lower=0.15)
# Final constraint to keep TE thickness at original or greater
DVCon.addThicknessConstraints1D(ptList=teList, nCon=2, axis=[0, 1, 0], lower=1.0, scaled=True)

if comm.rank == 0:
    fileName = os.path.join(args.output, "constraints.dat")
    DVCon.writeTecplot(fileName)
# rst dvcon (end)
# ======================================================================
#         Mesh Warping Set-up
# ======================================================================
# rst warp (beg)
meshOptions = {"gridFile": args.gridFile}

mesh = USMesh(options=meshOptions, comm=comm)
CFDSolver.setMesh(mesh)


# rst warp (end)
# ======================================================================
#         Functions:
# ======================================================================
# rst funcs (beg)
def cruiseFuncs(x):
    if MPI.COMM_WORLD.rank == 0:
        print(x)
    # Set design vars
    DVGeo.setDesignVars(x)
    ap.setDesignVars(x)
    # Run CFD
    CFDSolver(ap)
    # Evaluate functions
    funcs = {}
    DVCon.evalFunctions(funcs)
    CFDSolver.evalFunctions(ap, funcs)
    CFDSolver.checkSolutionFailure(ap, funcs)
    if MPI.COMM_WORLD.rank == 0:
        print(funcs)
    return funcs


def cruiseFuncsSens(x, funcs):
    funcsSens = {}
    DVCon.evalFunctionsSens(funcsSens)
    CFDSolver.evalFunctionsSens(ap, funcsSens)
    CFDSolver.checkAdjointFailure(ap, funcsSens)
    if MPI.COMM_WORLD.rank == 0:
        print(funcsSens)
        # write the sensitivities wrt surface coordinates to a file
        # iter_no = CFDSolver.adflow.iteration.itertot
        # fileName = os.path.join(args.output, "sens_%d.dat" % iter_no)
        # CFDSolver.writeSurfaceSensitivity(fileName, 'cd')
    return funcsSens


def objCon(funcs, printOK):
    # Assemble the objective and any additional constraints:
    funcs["obj"] = funcs[ap["cd"]]
    funcs["cl_con_" + ap.name] = funcs[ap["cl"]] - mycl
    if printOK:
        print("funcs in obj:", funcs)
    return funcs


# rst funcs (end)
# ======================================================================
#         Optimization Problem Set-up
# ======================================================================
# rst optprob (beg)
# Create optimization problem
optProb = Optimization("opt", MP.obj, comm=MPI.COMM_WORLD)

# Add objective
optProb.addObj("obj", scale=1e4)

# Add variables from the AeroProblem
ap.addVariablesPyOpt(optProb)

# Add DVGeo variables
DVGeo.addVariablesPyOpt(optProb)

# Add constraints
DVCon.addConstraintsPyOpt(optProb)
optProb.addCon("cl_con_" + ap.name, lower=0.0, upper=0.0, scale=1.0)

# The MP object needs the 'obj' and 'sens' function for each proc set,
# the optimization problem and what the objcon function is:
MP.setProcSetObjFunc("cruise", cruiseFuncs)
MP.setProcSetSensFunc("cruise", cruiseFuncsSens)
MP.setObjCon(objCon)
MP.setOptProb(optProb)
optProb.printSparsity()
# rst optprob (end)
# rst optimizer
# Set up optimizer
if args.opt.upper() == "SLSQP":
    optOptions = {"IFILE": os.path.join(args.output, "SLSQP.out")}
elif args.opt.upper() == "PSQP":
    optOptions = {
        "XMAX": 1.5,  # Maximum step size default: 1e16
        "TOLX": 1e-6,  # Variable change tolerance default: 1e-16
        "TOLC": 1e-4,  # Constraint violation tolerance default: 1e-6
        "TOLG": 1e-4,  # Lagrangian gradient tolerance default: 1e-6
        "RPF": 0.0001,  # Penalty coefficient default: 0.0001
        "MIT": 1000,  # Maximum number of iterations default: 1000
        "MFV": 2000,  # Maximum number of function evaluations default: 2000
        "MET": 2,  # Variable Metric Update (1 - BFGS, 2 - Hoshino) default: 2
        "MEC": 2,  # Negative Curvature Correction (1 - None, 2 - Powell’s Correction) default: 2
        "IFILE": os.path.join(args.output, "PSQP.out"),
    }
elif args.opt.upper() == "IPOPT":
    optOptions = {
        "max_iter": 200,
        "constr_viol_tol": 1e-6,
        # "nlp_scaling_method": "gradient-based",
        # "mu_init": 1e-1,
        "acceptable_tol": 1e-6,
        "acceptable_iter": 0,
        "tol": 1e-6,
        # "nlp_scaling_method": "none",
        "print_level": 0,
        "output_file": os.path.join(args.output, "IPOPT.out"),
        "file_print_level": 5,
        # "mu_strategy": "adaptive",
        "limited_memory_max_history": 50,
        # "corrector_type": "primal-dual",
        "print_user_options": "yes",
    }
elif args.opt.upper() == "SNOPT":
    optOptions = {
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-6,
        "Verify level": 0,  # -1,
        "Major iterations limit": 200,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        "Major step limit": 0.1,
        "Nonderivative linesearch": None,
        "Linesearch tolerance": 0.9,
        "Difference interval": 1e-6,
        "Function precision": 1e-8,
        "New superbasics limit": 2000,
        # "Penalty parameter": 1e-2,
        "Scale option": 1,
        "Hessian updates": 50,
        "Print file": os.path.join(args.output, "SNOPT_print.out"),
        "Summary file": os.path.join(args.output, "SNOPT_summary.out"),
    }
elif args.opt.upper() == "PAROPT":
    optOptions = {
        "algorithm": "tr",
        "tr_output_file": os.path.join(args.output, "paropt.tr"),  # Trust region output file
        "output_file": os.path.join(args.output, "paropt.out"),  # Interior point output file
        "tr_max_iterations": 200,  # Maximum number of trust region iterations
        "tr_infeas_tol": 1e-6,  # Feasibility tolerace
        "tr_l1_tol": 1e-5,  # l1 norm for the KKT conditions
        "tr_linfty_tol": 1e-5,  # l-infinity norm for the KKT conditions
        "tr_init_size": 0.001,  # Initial trust region radius
        "tr_min_size": 1e-6,  # Minimum trust region radius size
        "tr_max_size": 10.0,  # Max trust region radius size
        "tr_eta": 0.85,  # Trust region step acceptance ratio
        "tr_adaptive_gamma_update": True,  # Use an adaptive update strategy for the penalty
        "max_major_iters": 100,  # Maximum number of iterations for the IP subproblem solver
        "qn_subspace_size": 10,  # Subspace size for the quasi-Newton method
        "qn_type": "bfgs",  # Type of quasi-Newton Hessian approximation
        "abs_res_tol": 1e-8,  # Tolerance for the subproblem
        "starting_point_strategy": "affine_step",  # Starting point strategy for the IP
        "barrier_strategy": "mehrotra",  # Barrier strategy for the IP
        "use_line_search": False,  # Don't useline searches for the subproblem
    }
optOptions.update(args.optOptions)
opt = OPT(args.opt.upper(), options=optOptions)

# Run Optimization
sol = opt(optProb, MP.sens,  sensMode='pgc', sensStep=1e-6, storeHistory=os.path.join(args.output, "opt.hst"), storeSens=True)
if MPI.COMM_WORLD.rank == 0:
    print(sol)
