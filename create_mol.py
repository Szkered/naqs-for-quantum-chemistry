import pickle
import time

from pathlib import Path
import numpy as np
import openfermion as of
import openfermionpsi4 as ofpsi4
from absl import app, flags, logging
from openfermion.chem import MolecularData, geometry_from_pubchem
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import  get_sparse_operator

FLAGS = flags.FLAGS
flags.DEFINE_enum("cmd", "prepare", ["prepare", "create"], "which command to run")
flags.DEFINE_string("mol", "LiH", "molecule name")
# flags.DEFINE_string("basis", "sto-3g", "basis set to use")
flags.DEFINE_string("basis", "aug-cc-pCV5Z", "basis set to use")

MOLECULE_LIST = [
  "H2", "F2", "HCl", "LiH", "H2O", "CH2", "O2", "BeH2", "H2S", "NH3", "N2",
  "CH4", "C2", "LiF", "PH3", "LiCL", "Li2O", 'bicbut'
]


def get_geometry(molecule_name, verbose=True):
  if verbose and (molecule_name not in MOLECULE_LIST):
    print(
      f"Warning: {molecule_name} is not one of the molecules used in the paper"
      +
      "- that's not wrong, but just know it's not recreating the published results!"
    )

  # NOTE: handles geometries not in PubChem
  if molecule_name == "C2":
    # C2 isn't in PubChem - don't know why.
    geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])]
  elif molecule_name == 'bicbut':
    geometry = [
      ['C', (1.0487346562, 0.5208579773, 0.2375867187)],
      ['C', (0.2497284256, -0.7666691493, 0.0936474818)],
      ['C', (-0.1817326465, 0.4922777820, -0.6579637266)],
      ['C', (-1.1430708301, -0.1901383337, 0.3048494250)],
      ['H', (2.0107137141, 0.5520589541, -0.2623459977)],
      ['H', (1.0071921280, 1.0672669240, 1.1766131856)],
      ['H', (0.5438033167, -1.7129829738, -0.3260782874)],
      ['H', (-0.2580605320, 0.6268443026, -1.7229636111)],
      ['H', (-1.3778676954, 0.2935640723, 1.2498189977)],
      ['H', (-1.9664163102, -0.7380906148, -0.1402911727)]
    ]
  else:
    if molecule_name == "Li2O":
      # Li2O returns a different molecule - again, don't know why.
      molecule_name = "Lithium Oxide"
    geometry = geometry_from_pubchem(molecule_name)

  return geometry


def prepare_psi4(
  molecule_name, geometry=None, multiplicity=None, charge=None, basis=None
):

  if multiplicity is None:
    multiplicity = 1 if molecule_name not in ["O2", "CH2"] else 3
  if charge is None:
    charge = 0
  if basis is None:
    basis = 'sto-3g'

  if multiplicity == 1:
    reference = 'rhf'
    guess = 'sad'
  else:
    reference = 'rohf'
    guess = 'gwh'

  if geometry is None:
    geometry = get_geometry(molecule_name)

  geo_str = ""
  for atom, coords in geometry:
    geo_str += f"\n\t{atom}"
    for ord in coords:
      geo_str += f" {ord}"
    geo_str += ""

  psi4_str = f'''
molecule {molecule_name} {{{geo_str}
    {charge} {multiplicity}
    symmetry c1
}}
set basis       {basis}
set reference   {reference}

set globals {{
    basis {basis}
    freeze_core false
    fail_on_maxiter true
    df_scf_guess false
    opdm true
    tpdm true
    soscf false
    scf_type pk
    maxiter 1e6
    num_amps_print 1e6
    r_convergence 1e-6
    d_convergence 1e-6
    e_convergence 1e-6
    ints_tolerance EQUALITY_TOLERANCE
    damping_percentage 0
}}

hf = energy("scf")

# cisd = energy("cisd")
#ccsd = energy("ccsd")
ccsdt = energy("ccsd(t)")
fci = energy("fci")

print("Results for {molecule_name}.dat\\n")

print("""Geometry : {geo_str}\\n""")

print("HF : %10.6f" % hf)
# print("CISD : %10.6f" % cisd)
print("CCSD : %10.6f" % ccsd)
print("CCSD(T) : %10.6f" % ccsdt)
print("FCI : %10.6f" % fci)
    '''

  fname = f'{molecule_name}.dat'
  with open(fname, 'w+') as psi4_file:
    psi4_file.write(psi4_str)
  print(f"Created {fname}.")
  print(f"To solve molecule, run 'psi4 {fname}' from command line.")


def create_molecule_data(
  molecule_name,
  geometry=None,
  multiplicity=None,
  charge=None,
  basis=None,
  save_name=None
):

  if multiplicity is None:
    multiplicity = 1 if molecule_name not in ["O2", "CH2"] else 3
  if charge is None:
    charge = 0
  if basis is None:
    basis = 'sto-3g'
  if save_name is None:
    save_name = molecule_name

  if geometry is None:
    geometry = get_geometry(molecule_name)

  molecule = MolecularData(
    geometry,
    basis=basis,
    multiplicity=multiplicity,
    charge=charge,
    filename=save_name
  )

  # 1. Solve molecule and print results.

  print("Solving molecule with psi4", end="...")
  t_start = time.time()

  molecule = ofpsi4.run_psi4(
    molecule,
    run_scf=True,
    run_mp2=True,
    run_cisd=True,
    run_ccsd=True,
    run_fci=True,
    memory=16000,
    delete_input=True,
    delete_output=True,
    verbose=True
  )
  print("done in {:.2f} seconds".format(time.time() - t_start))

  print(f'{molecule_name} has:')
  print(f'\tgeometry of {molecule.geometry},')
  print(
    f'\t{molecule.n_electrons} electrons in {2*molecule.n_orbitals} spin-orbitals,'
  )
  print(f'\tHartree-Fock energy of {molecule.hf_energy:.6f} Hartree,')
  print(f'\tCISD energy of {molecule.cisd_energy:.6f} Hartree,')
  print(f'\tCCSD energy of {molecule.ccsd_energy:.6f} Hartree,')
  print(f'\tFCI energy of {molecule.fci_energy:.6f} Hartree.')

  # 2. Save molecule.

  # molecule.filename=save_name
  molecule.save()

  print(f"Molecule saved to {save_name}.hdf.")

  # 3. Convert molecular Hamiltonian to qubit Hamiltonian.
  print("Converting molecular Hamiltonian to qubit Hamiltonian", end="...")

  active_space_start = 0
  active_space_stop = molecule.n_orbitals

  # Get the Hamiltonian in an active space.
  molecular_hamiltonian = molecule.get_molecular_hamiltonian(
    occupied_indices=None,
    active_indices=range(active_space_start, active_space_stop)
  )

  fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
  qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
  qubit_hamiltonian.compress()

  print("done in {:.2f} seconds".format(time.time() - t_start))

  # 3. Save qubit Hamiltonian.

  with open(save_name + "_qubit_hamiltonian.pkl", 'wb') as f:
    pickle.dump(qubit_hamiltonian, f)

  print(f"Qubit Hamiltonian saved to {save_name+'_qubit_hamiltonian.pkl'}.")


def main(_):
  if FLAGS.cmd == "prepare":
    prepare_psi4(FLAGS.mol, basis=FLAGS.basis)
  elif FLAGS.cmd == "create":
    create_molecule_data(FLAGS.mol)


if __name__ == '__main__':
  app.run(main)
