from ase import Atoms
import ase
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from matdeeplearn.common.ase_utils import MDLCalculator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../.."))))
from msp.utils.objectives import Energy
from msp.forcefield import MDL_FF
import json
from msp.dataset import download_dataset


class PSO():
    def __init__(self):
        self.composition = [22, 22, 8, 8, 8, 8]
        self.cell = [5.56641437, 5.56641437, 5.56641437, 140.05412510, 140.05412510, 57.77112661]
        train_config = 'mdl_config.yml'
        self.calculator = MDLCalculator(config=train_config)
        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []

    def init_atoms(self, density=0.2):
        atoms = Atoms(self.composition, cell=self.cell, pbc=(True, True, True))

        scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        atoms.set_scaled_positions(scaled_positions)

        return atoms

    def dimensions_to_atoms(self, params):
        positions = params.reshape(-1, 3)
        atoms = Atoms(self.composition, cell=self.cell, pbc=(True, True, True), positions=positions)
        return atoms

    def objective_func(self, params):
        atoms = self.dimensions_to_atoms(params)

        atoms.set_calculator(self.calculator)
        loss = atoms.get_potential_energy()

        if loss < self.best_loss:
            self.best_loss = loss

        return loss

    def f(self, x):
        """Higher-level method to do forward_prop in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [self.objective_func(x[i]) for i in range(n_particles)]

        self.best_losses.append(self.best_loss)
        self.avg_losses.append(np.mean(j))

        return np.array(j)

    def run(self):
        for i in range(5):
            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} # cognitive, social, itertia
            particles = 10 # number of particles in system
            dimensions = len(self.composition)*3 # first 9 are cell, rest are atom positions

            init_positions = np.empty((particles, dimensions))
            for i in range(particles):
                init_atoms = self.init_atoms()
                init_pos = [float(i) for l in init_atoms.positions for i in l]
                init_positions[i] = np.array(init_pos)
            filename = f"init_structure_pso_{i+1}" + ".cif"
            ase.io.write(filename, self.dimensions_to_atoms(init_positions[0]))

            optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions)

            # Perform optimization
            cost, pos = optimizer.optimize(self.f, iters=10000)

            filename = f"best_structure_{i+1}" + ".cif"
            ase.io.write(filename, self.dimensions_to_atoms(pos))

            plt.plot(self.best_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Best Loss')
            plt.title('Best Losses')
            plt.savefig(f'best_losses_pso_{i+1}'+'.png')
            plt.close()

            plt.plot(self.avg_losses)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title('Average Losses')
            plt.savefig(f'avg_losses_pso_{i+1}'+'.png')
            plt.close()

if __name__ == "__main__":
    pso = PSO()
    pso.run()