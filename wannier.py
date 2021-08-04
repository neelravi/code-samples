from math import pi
from pickle import load, dump

import numpy as np
from ase.units import Bohr
import sys

from _gpaw import localize
from gpaw.utilities.tools import dagger, lowdin

from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional
from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.utilities import pack, unpack
from gpaw.lfc import LFC
import _gpaw
from gpaw.paw import PAW 
from gpaw.overlap import Overlap
from gpaw.hs_operators import MatrixOperator
from gpaw.kohnsham_layouts import get_KohnSham_layouts_ravindra
from gpaw.utilities import hartree
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from ase.units import Hartree
from gpaw.xc.sic import SIC
from gpaw.io import Reader, Writer

np.set_printoptions(threshold=sys.maxsize)


class Wannier:
    def __init__(self, 
                calc, 
                kpt=None,
                paw=None,
                density=None,
                hamiltonian=None,
                wfs=None,
                poissonsolver=None,
                ghat=None,
                spin=0, 
                nbands=None,
                dtype=float,
                coulomb_factor=0.5,
                xc_factor=0.5):
        """ 
        Developer: Ravindra Shinde, UCR.
        Project  : FLOSIC for periodic systems

        A class to obtain the Wannier functions using Kohn-Sham wavefunctions. 
        Note that the wannier functions obtained here are not maximally localized. 
        They carry a gauge freedom.

        This class is being used to calculate the self-interaction energies and to plot
        the SIC corrected bandstructures.

        Important Variables
        coulomb_factor:
            Scaling factor for Hartree-functional

        xc_factor:
            Scaling factor for xc-functional

        """

        self.Z_nnc = None
        self.S_unn = None           # Contains overlap between states for all the kpoints
        self.wfs = calc.wfs
        self.kpt = calc.wfs.kpt_u
        # self.xc = calc.wfs.xc
        self.poissonsolver = poissonsolver
        self.ghat = ghat
        self.pt = calc.wfs.pt
        self.gd = calc.wfs.gd

        self.nspins = calc.wfs.nspins
        self.spin = spin #kpt.s
        self.timer = calc.wfs.timer
        self.setups = calc.wfs.setups

        self.dtype = dtype
        self.coulomb_factor = coulomb_factor
        self.xc_factor = xc_factor

        self.nocc = None  # number of occupied states
        self.W_mn = None  # unit. transf. to energy optimal states
        self.initial_W_mn = None  # initial unitary transformation
        self.vt_mG = None  # orbital dependent potentials
        self.exc_m = None  # SIC contributions (from E_xc)
        self.ecoulomb_m = None  # SIC contributions (from E_H)

        if calc is not None:
            if not calc.wfs.gd.orthogonal:
                raise NotImplementedError('Wannier function analysis ' +
                                          'requires an orthogonal cell.')

            self.cell_c = calc.wfs.gd.cell_cv.diagonal() * Bohr
            print ("printing cell_c", self.cell_c)
            if nbands is None:
                nbands = calc.get_number_of_bands()
            self.Z_nnc = np.empty((nbands, nbands, 3), complex)

            print("calculating Z_nnc")
            for c in range(3):
                G_c = np.zeros(3)
                G_c[c] = 1
                self.Z_nnc[:, :, c] = calc.get_wannier_integrals(
                    spin, 0, 0, G_c, nbands)
            self.value = 0.0
            self.U_nn = np.identity(nbands)

    @staticmethod
    def ismaxlocalized():
        print("Caution:: The Wannier functions are not maximally localized")


    @property
    def sic_factors(self):
        return "SIC Coulomb factor = {:>2}, SIC XC factor      = {:>2}".format(self.coulomb_factor, self.xc_factor)


    def load(self, filename):
        self.cell_c, self.Z_nnc, self.value, self.U_nn = load(open(filename))

    def dump(self, filename):
        dump((self.cell_c, self.Z_nnc, self.value, self.U_nn), filename)
        
    def localize(self, eps=1e-5, iterations=-1):

        i = 0
        print("iteration , Maximizaton")
        while i != iterations:
            value = localize(self.Z_nnc, self.U_nn)
            print(i, value)

            if value - self.value < eps:
                break
            i += 1
            self.value = value
#        print ("localizing matrix U_nn", self.U_nn)
        return value # / Bohr**6

    def get_centers(self):
        scaled_c = -np.angle(self.Z_nnc.diagonal()).T / (2 * pi)
        print ( "centers from wannier routine", (scaled_c % 1.0) * self.cell_c )
        return (scaled_c % 1.0) * self.cell_c        



    def calculate_energies(self, calc, centers):
        ESIC = 0
        xc = calc.hamiltonian.xc
        assert xc.type == 'GGA'         # Considering only PBE exchange-correlation functionals

        # Calculate the energy contribution from the core orbitals
        print('Atomic core energies for the atoms')
        print('{}'.format('--------------------------------------------------------------------------------'))
        print('{:>10} {:>15} {:>15} {:>15} {:>15}'.format('#atom & Core', 'E_xc[core]', 'E_coul[core]', 'E[core]', 'E[cumulative]'))
        print('{}'.format('--------------------------------------------------------------------------------'))
        for a in calc.density.D_asp:
            setup = calc.density.setups[a]
            g = Generator(setup.symbol, xcname='PBE', nofiles=True, txt=None)
            g.run(**parameters[setup.symbol])
            njcore = g.njcore
            cumulative_f = 0
            for f, l, e, u in zip(g.f_j[:njcore], g.l_j[:njcore],
                                  g.e_j[:njcore], g.u_j[:njcore]):
                na = np.where(abs(u) < 1e-160, 0,u)**2 / (4 * pi)
                cumulative_f += f
                na[1:] /= g.r[1:]**2
                na[0] = na[1]
                nb = np.zeros(g.N)
                v_sg = np.zeros((2, g.N))
                vHr = np.zeros(g.N)
                Exc = xc.calculate_spherical(g.rgd, np.array([na, nb]), v_sg)
                hartree(0, na * g.r * g.dr, g.r, vHr)
                EHa = 2*pi*np.dot(vHr*na*g.r , g.dr)
                ESIC += -f*(EHa+Exc)
                self.pretty_print_core_energies(setup.symbol, a, f, cumulative_f, l, e, Exc, EHa, ESIC)
            print('{}'.format('****'))

        print('{}'.format('--------------------------------------------------------------------------------'))
        sic = SIC(xc='PBE', finegrid=True, coulomb_factor=0.5, xc_factor=0.5)
        print (SIC.__doc__)
        sic.initialize_flosic(calc.density, calc.hamiltonian, calc.wfs)       
        sic.set_positions(calc.spos_ac)

        ESIC2 = 0.0
        print('{}'.format('---------------------------------------------------------------------'))
        print('Valence electron self-interaction correction contribution')
        print('{}'.format('---------------------------------------------------------------------'))        
        print('{:^10} {:^10} {:^10}  {:^10}  {:^10} '.format('spin', 'band', 'E_xc', 'E_coul', 'Esic'))        
        print('{}'.format('---------------------------------------------------------------------'))        

#        wfl_density = self.wfs.gd.empty(self.nspins) 
#        wfl_density.fill(0.0)


        for s, spin in sic.spin_s.items():
            spin.initialize_orbitals(calc)             # construction of unitary matrix for wanniers
            spin.update_optimal_states_kpts(calc, centers)      # multiplication of unitary matrix to wannier functions
            spin.update_potentials(calc)           # the sic potential and the energies are evaluated here
        # ham = calc.hamiltonian
        # ham.update(wfl_density)
        # ham.get_energy(calc.occupations)
        n = 0
        for xc, c in zip(spin.exc_m, spin.ecoulomb_m):              
            print('{:^10d} {:^10d} {:^10.6f} {:^10.6f} {:^10.6f} '.format(s, n, -xc, -c, -2*(xc + c)))                        
            n += 1

        ESIC2 += spin.esic
        print('{}'.format('---------------------------------------------------------------------'))            
        print('Self-Interaction corrected total energy:')
        dft_energy = (calc.get_potential_energy())
        total = (ESIC2 + calc.get_potential_energy()) #+ calc.get_reference_energy())
        print('{:>16} {}  {}  {:>16} {} {} {:>16} {}'.format('DFT energy ' , 'eV', ' + ', ' SIC ', 'eV', ' = ',  ' Total', 'eV'))                   
        print('{:>16.6f} {}  {}  {:>16.6f} {} {} {:>16.6f} {}'.format(dft_energy, 'eV', ' + ', ESIC2, 'eV', ' = ',  total, 'eV'))                   
        print('{}'.format('---------------------------------------------------------------------'))                    
        return total


    def pretty_print_core_energies(self, symbol, a, f, cumulative_f, l, e, Exc, EHa, ESIC):
        angular   = ["s", "p", "d"]

        if cumulative_f == 2:
            principal = "1"
        elif cumulative_f >= 4:
            principal = "2"
        elif cumulative_f > 10:
            principal = "3"
        else:
            raise NotImplementedError('periodic-FLOSIC for d-orbital elements not implemented.')

        electronic_term =  (principal+angular[l]+str(f))
        print('{:>3}{}{:>2}{} {} {:15.6f} {:15.6f} {:15.6f} {:15.6f}'.format(symbol,'[',str(a),']', electronic_term, Exc, EHa, -f*(EHa+Exc), ESIC))



    def get_function(self, calc, u, n, pad=True):
        print (' inside get_function kpts', u)
        if pad:
            return calc.wfs.gd.zero_pad(self.get_function(calc, u , n, False))
        psit_nG = calc.wfs.kpt_u[u].psit_nG
        psit_nG = psit_nG.reshape((calc.wfs.bd.nbands, -1))
        A = np.dot(self.U_nn[:, n], psit_nG).reshape(calc.wfs.gd.n_c) / Bohr**1.5
        return A 

    def get_function_finegrid(self, calc, n, pad=True):
        if pad:
            return calc.wfs.gd.zero_pad(self.get_function(calc, n, False))
        psit_ng = calc.wfs.kpt_u[self.spin].psit_ng[:]
        psit_ng = psit_ng.reshape((calc.wfs.bd.nbands, -1))
        A = np.dot(self.U_nn[:, n], psit_ng).reshape(calc.wfs.gd.n_c) / Bohr**1.5
        normalization = (calc.wfs.gd.dv*Bohr**3)*np.sum(A*A)
        return A/np.sqrt(normalization)

    def get_kohn_sham_psi(self, calc, n, pad=True):
        if pad:
            return calc.wfs.gd.zero_pad(self.get_function(calc, n, False))
        nbands = calc.get_number_of_bands()            
        psit_nG = calc.wfs.kpt_u[self.spin].psit_nG[:]
        psit_nG = psit_nG.reshape((calc.wfs.bd.nbands, -1))
        U_nn = np.identity(nbands)
        A = np.dot(U_nn[:, n], psit_nG).reshape(calc.wfs.gd.n_c) / Bohr**1.5
        return A


    def get_hamiltonian(self, calc):
        # U^T diag(eps_n) U
        eps_n = calc.get_eigenvalues(kpt=0, spin=self.spin)
        return np.dot(dagger(self.U_nn) * eps_n, self.U_nn)


class LocFun(Wannier):
    def localize(self, calc, M=None, T=0, projections=None, ortho=True,
                 verbose=False):
        # M is size of Hilbert space to fix. Default is ~ number of occ. bands.
        if M is None:
            M = 0
            f_n = calc.get_occupation_numbers(0, self.spin)
            while f_n[M] > .01:
                M += 1

        if projections is None:
            projections = single_zeta(calc, self.spin, verbose=verbose)
        
        self.U_nn, self.S_jj = get_locfun_rotation(projections, M, T, ortho)


        if self.Z_nnc is None:
            self.value = 1
            return self.value
        
        self.Z_jjc = np.empty(self.S_jj.shape + (3,))
        for c in range(3):
            self.Z_jjc[:, :, c] = np.dot(dagger(self.U_nn),
                                     np.dot(self.Z_nnc[:, :, c], self.U_nn))
        
        self.value = np.sum(np.abs(self.Z_jjc.diagonal())**2)

# two debug lines ravindra
        print("Printing Z_nnc in the localize function")            
        # print(self.Z_nnc)

        return self.value # / Bohr**6

    def get_centers(self):
        z_jjc = np.empty(self.S_jj.shape+(3,))
        for c in range(3):
            z_jjc = np.dot(dagger(self.U_nn),
                            np.dot(self.Z_nnc[:,:,c], self.U_nn))

        scaled_c = -np.angle(z_jjc.diagonal()).T / (2 * pi)
        return (scaled_c % 1.0) * self.cell_c

    def get_eigenstate_centers(self):
        scaled_c = -np.angle(self.Z_nnc.diagonal()).T / (2 * pi)
        return (scaled_c % 1.0) * self.cell_c
    
    def get_proj_norm(self, calc):
        return np.array([np.linalg.norm(U_j) for U_j in self.U_nn])
            
