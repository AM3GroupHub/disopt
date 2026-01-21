import sys
import os
import numpy as np
import copy
from time import time
import torch

# MACE imports
from mace.calculators import MACECalculator

# ASE imports
from ase import units
from ase.io import read, write
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

############# Parameter Settings #############
# Target Temperature 1273 K
temperature = 1273

# Force convergence criteria (eV/A)
relax_fmax = 0.05

# Max allowed steps for single FIRE relaxation
max_relax_steps = 200

# Total MC steps
steps_mc = 100000

# File paths
cuda_device = "cuda"
model_path = 'MACE-matpes-r2scan-omat-ft.model' # Please ensure the model file is in the current directory or specify the correct path
input_traj = 'LiMnTiO_222.xyz'   # Initial structure filename, please ensure it contains Li, Mn, Ti, O
output_file = 'fireswap_trajectory.xyz'
log_file = 'fireswap.log'

###################################

# 1. Load MACE model
print(f"Loading MACE model: {model_path} ...")
calculator = MACECalculator(
    model_paths=model_path,
    device=cuda_device,
    default_dtype="float64"  # Recommend using float64 to improve precision
)

# 2. Read initial structure
print(f"Reading initial structure: {input_traj}")
try:
    atoms = read(input_traj) # Default read the last frame
except Exception as e:
    print(f"Read failed: {e}")
    sys.exit(1)

atoms.set_calculator(calculator)

# 3. Define FIRE relaxation function
def run_fire_relaxation(atoms_obj, fmax, steps=None):
    """
    Perform FIRE geometric optimization.
    """
    # Use FrechetCellFilter to allow simultaneous relaxation of cell and atoms
    ucf = FrechetCellFilter(atoms_obj)
    
    # Initial structure has no step limit (or a large value), loop has limited steps
    if steps is None:
        steps = 10000
    
    # dt=0.1, maxstep=0.2 are empirical parameters
    opt = FIRE(ucf, logfile=None, dt=0.1, maxstep=0.2)
    
    try:
        opt.run(fmax=fmax, steps=steps)
        return True
    except Exception as e:
        print(f"Relaxation Error: {e}")
        return False

def get_energy_float(atoms_obj):
    e = atoms_obj.get_potential_energy()
    if isinstance(e, np.ndarray):
        return float(e.item()) if e.size == 1 else float(e.sum())
    return float(e)

# 4. Initialization
kT = temperature * units.kB

# Identify atom indices
symbols = np.array(atoms.get_chemical_symbols())
# Cation group: Li, Mn, Ti
indices_cation = [i for i, s in enumerate(symbols) if s in ['Li', 'Mn', 'Ti']]
# Anion group: O 
indices_anion = [i for i, s in enumerate(symbols) if s == 'O']

print(f"System contains: Cations {len(indices_cation)}, Anions {len(indices_anion)}")

if os.path.exists(log_file): os.remove(log_file)
if os.path.exists(output_file): os.remove(output_file)

with open(log_file, 'w') as f:
    f.write("Step,Energy(eV),Delta_E(eV),Accepted,Type,Time(s)\n")

# --- Stage A: Full relaxation of initial structure ---
print("\n=== Stage A: Initial structure full relaxation ===")
run_fire_relaxation(atoms, fmax=relax_fmax, steps=None)
e_current = get_energy_float(atoms)
write(output_file, atoms, format='extxyz', append=True)
print(f"  Initial energy: {e_current:.5f} eV")

# --- Stage B: FIRE-Swap Loop ---
print(f"\n=== Stage B: FIRE-Swap Loop (T={temperature}K) ===")

accept_count = 0

for step in range(1, steps_mc + 1):
    step_start = time()
    
    # 1. Backup (for rollback)
    old_positions = atoms.get_positions()
    old_cell = atoms.get_cell()
    old_symbols = atoms.get_chemical_symbols()
    
    # 2. Propose Swap 
    # === Modification: Fixed to cation swap ===
    swap_type = 'cation'
    idx1, idx2 = -1, -1

    max_retries = 1000
    found_distinct = False
    
    current_symbols = np.array(atoms.get_chemical_symbols())

    for _ in range(max_retries):
        # Randomly select two from Li, Mn, Ti
        idx1, idx2 = np.random.choice(indices_cation, 2, replace=False)
        
        # Break loop only if element types are different (e.g., Li-Mn)
        if current_symbols[idx1] != current_symbols[idx2]:
            found_distinct = True
            break
            
    if not found_distinct:
        print(f"Warning: Step {step} skipped, could not find distinct pair for {swap_type} swap.")
        continue

    # Perform swap
    current_symbols[idx1], current_symbols[idx2] = current_symbols[idx2], current_symbols[idx1]
    atoms.set_chemical_symbols(current_symbols)
    
    # 3. Geometric relaxation (limited to 100 steps)
    run_fire_relaxation(atoms, fmax=relax_fmax, steps=max_relax_steps)
    
    # 4. Calculate energy difference
    e_trial = get_energy_float(atoms)
    delta_e = e_trial - e_current
    
    # 5. Metropolis criterion
    if delta_e < 0:
        accepted = True
    else:
        p = np.exp(-delta_e / kT)
        accepted = (np.random.rand() < p)
        
    status = "ACC" if accepted else "REJ"

    # 6. Update or rollback
    if accepted:
        e_current = e_trial
        accept_count += 1
        # Structure updated, no action needed
    else:
        # Rejected: Rollback R, cell, and symbols
        atoms.set_cell(old_cell)
        atoms.set_positions(old_positions)
        atoms.set_chemical_symbols(old_symbols)
        # e_current remains unchanged

    # 7. Logging
    step_time = time() - step_start
    
    with open(log_file, 'a') as f:
        f.write(f"{step},{e_current:.5f}    {delta_e:.5f}    {accepted}    {swap_type}    {step_time:.2f}\n")
    
    print(f"Step {step}: {status} ({swap_type})    dE={delta_e:.4f}    E={e_current:.4f}    Time={step_time:.2f}s")

    # 8. Save structure (save every step)
    write(output_file, atoms, format='extxyz', append=True)

print("FIRE-Swap simulation completed.")