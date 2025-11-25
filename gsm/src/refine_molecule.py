"""
Molecule refinement using RDKit.

Refines predicted 3D coordinates to valid molecular structures.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist


class MoleculeRefiner:
    """
    Refines predicted 3D coordinates to valid molecular structures.
    
    Strategy:
    1. Create RDKit molecule from atom types
    2. Optionally guess bonds from distances
    3. Set predicted coordinates as conformer
    4. Run force field optimization to push to nearest valid geometry
    5. Verify chemical validity
    """
    
    def __init__(
        self,
        force_field="MMFF",
        max_iters=500,
        bond_distance_tolerance=0.3,  # Angstroms above covalent radius
        verbose=False,
    ):
        """
        Args:
            force_field: "MMFF" or "UFF"
            max_iters: Maximum iterations for force field optimization
            bond_distance_tolerance: Tolerance for guessing bonds (Angstroms)
            verbose: Print detailed information
        """
        self.force_field = force_field.upper()
        self.max_iters = max_iters
        self.bond_distance_tolerance = bond_distance_tolerance
        self.verbose = verbose
        
        # Covalent radii for common atoms (Angstroms)
        # Source: Cordero et al. Dalton Trans. 2008
        self.covalent_radii = {
            'H': 0.31, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
            'F': 0.57, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
            'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'I': 1.39,
        }
    
    def estimate_bond_orders_from_coords(self, coords, elements):
        """
        Estimate bond orders from atomic distances.
        
        Args:
            coords: [N, 3] atomic coordinates
            elements: [N] element symbols
            
        Returns:
            List of (i, j, bond_order) tuples
        """
        n_atoms = len(coords)
        distances = cdist(coords, coords)
        
        bonds = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Get covalent radii
                r_i = self.covalent_radii.get(elements[i], 1.0)
                r_j = self.covalent_radii.get(elements[j], 1.0)
                
                # Expected single bond distance
                expected_dist = r_i + r_j
                actual_dist = distances[i, j]
                
                # Check if within bonding distance
                if actual_dist < expected_dist + self.bond_distance_tolerance:
                    # Estimate bond order from distance
                    # Shorter = higher order (rough heuristic)
                    if actual_dist < expected_dist * 0.85:
                        bond_order = 3  # Triple
                    elif actual_dist < expected_dist * 0.92:
                        bond_order = 2  # Double
                    else:
                        bond_order = 1  # Single
                    
                    bonds.append((i, j, bond_order))
        
        return bonds
    
    def create_molecule_from_atoms(self, elements, guess_bonds=False, coords=None):
        """
        Create RDKit molecule from atom types.
        
        Args:
            elements: List of element symbols
            guess_bonds: Whether to guess bonds from coordinates
            coords: [N, 3] coordinates (required if guess_bonds=True)
            
        Returns:
            RDKit Mol object or None if failed
        """
        # Create editable molecule
        mol = Chem.RWMol()
        
        # Add atoms
        atom_indices = []
        for elem in elements:
            atom = Chem.Atom(elem)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)
        
        if guess_bonds and coords is not None:
            # Estimate bonds from coordinates
            bonds = self.estimate_bond_orders_from_coords(coords, elements)
            
            for i, j, bond_order in bonds:
                if bond_order == 1:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                elif bond_order == 2:
                    mol.AddBond(i, j, Chem.BondType.DOUBLE)
                elif bond_order == 3:
                    mol.AddBond(i, j, Chem.BondType.TRIPLE)
        
        # Convert to Mol
        mol = mol.GetMol()
        
        # Sanitize (check chemical validity)
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            if self.verbose:
                print("  Warning: Could not sanitize molecule, trying without sanitization")
            return None
    
    def set_conformer_coords(self, mol, coords):
        """
        Set 3D coordinates as conformer.
        
        Args:
            mol: RDKit Mol object
            coords: [N, 3] numpy array of coordinates
            
        Returns:
            Mol with conformer or None if failed
        """
        try:
            conformer = Chem.Conformer(mol.GetNumAtoms())
            
            for i, (x, y, z) in enumerate(coords):
                conformer.SetAtomPosition(i, (float(x), float(y), float(z)))
            
            mol.AddConformer(conformer, assignId=True)
            return mol
        except Exception as e:
            if self.verbose:
                print(f"  Error setting conformer: {e}")
            return None
    
    def optimize_geometry(self, mol):
        """
        Optimize molecular geometry using force field.
        
        Args:
            mol: RDKit Mol object with conformer
            
        Returns:
            (optimized_mol, energy, converged)
        """
        try:
            if self.force_field == "MMFF":
                # Try MMFF first
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol,
                    AllChem.MMFFGetMoleculeProperties(mol),
                    confId=0
                )
                if ff is None:
                    # Fallback to UFF
                    if self.verbose:
                        print("  MMFF failed, falling back to UFF")
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
            
            if ff is None:
                return mol, None, False
            
            # Optimize
            converged = ff.Minimize(maxIts=self.max_iters)
            energy = ff.CalcEnergy()
            
            return mol, energy, (converged == 0)
            
        except Exception as e:
            if self.verbose:
                print(f"  Error during optimization: {e}")
            return mol, None, False
    
    def get_coords_from_mol(self, mol):
        """
        Extract coordinates from RDKit molecule.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            [N, 3] numpy array
        """
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        return np.array(coords)
    
    def validate_molecule(self, mol):
        """
        Check if molecule is chemically valid.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # Check valence
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            issues.append(f"Sanitization failed: {e}")
        
        # Check for disconnected fragments
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            issues.append(f"Molecule has {len(frags)} disconnected fragments")
        
        # Check for radicals (unpaired electrons)
        for atom in mol.GetAtoms():
            num_radical_electrons = atom.GetNumRadicalElectrons()
            if num_radical_electrons > 0:
                issues.append(f"Atom {atom.GetIdx()} has {num_radical_electrons} unpaired electrons")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def refine_to_molecule(
        self,
        coords,
        elements,
        guess_bonds=True,
    ):
        """
        Main refinement function.
        
        Args:
            coords: [N, 3] predicted coordinates
            elements: [N] element symbols
            guess_bonds: Whether to guess bonds from distances
            
        Returns:
            (mol, refined_coords, success)
            - mol: RDKit Mol object (or None if failed)
            - refined_coords: [N, 3] refined coordinates
            - success: Whether refinement succeeded
        """
        if self.verbose:
            print(f"Refining molecule with {len(coords)} atoms...")
        
        # Create molecule
        mol = self.create_molecule_from_atoms(
            elements,
            guess_bonds=guess_bonds,
            coords=coords if guess_bonds else None,
        )
        
        if mol is None:
            if self.verbose:
                print("  Failed to create molecule")
            return None, coords, False
        
        if self.verbose:
            print(f"  Created molecule with {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
        
        # Set initial coordinates
        mol = self.set_conformer_coords(mol, coords)
        if mol is None:
            return None, coords, False
        
        # Optimize geometry
        mol, energy, converged = self.optimize_geometry(mol)
        
        if self.verbose:
            status = "converged" if converged else "did not converge"
            print(f"  Optimization {status}, energy: {energy if energy else 'N/A'}")
        
        # Extract refined coordinates
        refined_coords = self.get_coords_from_mol(mol)
        
        # Validate
        is_valid, issues = self.validate_molecule(mol)
        
        if self.verbose:
            if is_valid:
                print("  ✓ Molecule is chemically valid")
            else:
                print(f"  ✗ Validation issues: {'; '.join(issues)}")
        
        success = is_valid and converged
        
        return mol, refined_coords, success
    
    def refine_batch(self, coords_batch, elements_batch, guess_bonds=True):
        """
        Refine a batch of molecules.
        
        Args:
            coords_batch: List of [N, 3] coordinate arrays
            elements_batch: List of element symbol lists
            guess_bonds: Whether to guess bonds
            
        Returns:
            List of (mol, refined_coords, success) tuples
        """
        results = []
        for coords, elements in zip(coords_batch, elements_batch):
            result = self.refine_to_molecule(coords, elements, guess_bonds)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test the refiner
    print("Testing MoleculeRefiner...")
    print("=" * 80)
    
    # Create a simple molecule: ethanol (CCO)
    # Ground truth SMILES: CCO
    elements = ['C', 'C', 'O']
    
    # Ideal coordinates (from RDKit)
    coords_ideal = np.array([
        [0.0000,  0.0000,  0.0000],  # C
        [1.5200,  0.0000,  0.0000],  # C
        [2.0200,  1.3700,  0.0000],  # O
    ])
    
    # Add some noise to simulate prediction
    coords_noisy = coords_ideal + np.random.randn(*coords_ideal.shape) * 0.3
    
    print(f"\nTest case: Ethanol (CCO)")
    print(f"  Ground truth coords:\n{coords_ideal}")
    print(f"  Noisy coords:\n{coords_noisy}")
    
    # Test refinement
    refiner = MoleculeRefiner(verbose=True)
    
    mol, refined_coords, success = refiner.refine_to_molecule(
        coords_noisy,
        elements,
        guess_bonds=True,
    )
    
    print(f"\n  Refined coords:\n{refined_coords}")
    print(f"  Success: {success}")
    
    if mol is not None:
        # Get SMILES
        smiles = Chem.MolToSmiles(mol)
        print(f"  Generated SMILES: {smiles}")
        
        # Compute RMSD to ideal
        rmsd = np.sqrt(((refined_coords - coords_ideal) ** 2).mean())
        print(f"  RMSD to ideal: {rmsd:.4f} Å")
    
    print("\n" + "=" * 80)
    print("✓ Test complete!")
