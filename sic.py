# ============================================================================
#                            Important Information 
# ============================================================================
""" This module is being developed for the calculations of Fermi Lowdin orbital
self-interaction corrections to the DFT functionals. An earlier wannier-based 
implementation was modified heavily to incorporate this methodology.

The wannier functions are inherited from the GPAW's own wannier class. The Fermi
Lowdin functions are then constructed on top of the wanniers. The XC and Coulomb 
contributions are calculated and the total correction is minimized with respect to
the Wannier charge centers.

Developed by:
Dr. Ravindra Shinde (neelravi@gmail.com) University of California Riverside.
"""

from math import pi

import numpy as np
from ase.units import Bohr, Hartree
from ase.utils import basestring

from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc import XC
import copy
from gpaw.density import Density
from gpaw.xc.functional import XCFunctional
from gpaw.poisson import PoissonSolver
from gpaw.coulomb import Coulomb
from gpaw.transformers import Transformer
from gpaw.utilities import pack, unpack
from gpaw.lfc import LFC
from gpaw.utilities import (unpack2, unpack_atomic_matrices, pack_atomic_matrices)
from gpaw.io import Reader, Writer
import gpaw.mpi as mpi
import ase.io.ulm as ulm
import _gpaw




class SIC(XCFunctional):
    ''' 
    This is the main driver class used for the calculation of the SIC contribution from
    the Fermi Lowdin orbitals. Here we assume that the lattice vectors are orthogonal and 
    only GGA type of functionals are used.
 
    '''


    orbital_dependent = True
    unitary_invariant = False


    def __init__(self, xc='PBE', finegrid=False, **parameters):
        """Self-Interaction Correction 

        finegrid: boolean
            Use fine grid for final energy functional evaluations
        """

        if isinstance(xc, basestring):
            xc = XC(xc)

        if xc.orbital_dependent:
            raise ValueError('SIC does not support ' + xc.name)

        self.xc = xc
        XCFunctional.__init__(self, xc.name + '-PZ-SIC', xc.type)
        self.finegrid = finegrid

        self.parameters = parameters

    def __repr__(self):
        return "<SIC Class - Self Interaction corrections :: Functional = {0}, Orbital Dependent = {1}, Unitary Invariant:{2}>".format(self.xc, self.orbital_dependent, self.unitary_invariant)


    # Proper IO of general SIC objects should work by means of something like:
    def todict(self):
        return {
            'type': 'SIC',
            'name': self.xc.name,
            'finegrid': self.finegrid,
            'parameters': dict(self.parameters)}

    def initialize(self, density, hamiltonian, wfs, occ=None):

        assert wfs.bd.comm.size == 1  # band parallelization unsupported

        self.wfs = wfs
        self.dtype = float
        self.xc.initialize(density, hamiltonian, wfs, occ)
        self.kpt_comm = wfs.kd.comm
        self.nspins = wfs.nspins
        self.nbands = wfs.bd.nbands
        print ('number of bands = ',self.nbands)

        self.finegd = density.gd.refine() if self.finegrid else density.gd


        self.ghat = LFC(self.finegd,
                        [setup.ghat_l for setup in density.setups],
                        integral=np.sqrt(4 * np.pi),
                        forces=True)

        poissonsolver = PoissonSolver(eps=1e-14)
        poissonsolver.set_grid_descriptor(self.finegd)

        self.spin_s = {}
        for kpt in wfs.kpt_u:
            self.spin_s[kpt.s] = SICSpin(kpt, self.xc, density, hamiltonian,
                                         wfs, poissonsolver, self.ghat,
                                         self.finegd, **self.parameters)


    def initialize_flosic(self, density, hamiltonian, wfs, occ=None):
#        assert wfs.kd.gamma
        assert wfs.bd.comm.size == 1  # no parallelization to start with

        self.wfs = wfs
        self.dtype = float
        self.xc.initialize(density, hamiltonian, wfs, occ)
        self.kpt_comm = wfs.kd.comm
        self.nspins = wfs.nspins
        self.nbands = wfs.bd.nbands
        self.density = density
        self.U_nn = np.identity(self.nbands)
        # self.finegd = density.gd.refine() if self.finegrid else density.gd
        self.finegd = density.gd

        # print ('self finegrid in flosic = ',self.finegrid)

        self.ghat = LFC(self.finegd,
                        [setup.ghat_l for setup in density.setups],
                        integral=np.sqrt(4 * np.pi),
                        forces=True)

        poissonsolver = PoissonSolver(eps=1e-14)
        poissonsolver.set_grid_descriptor(self.finegd)

        self.spin_s = {}
        for kpt in wfs.kpt_u:
            self.spin_s[kpt.s] = SICSpin(kpt, self.xc, density, hamiltonian,
                                         wfs,  poissonsolver, self.ghat,
                                         self.finegd, **self.parameters)
        return


    def get_setup_name(self):
        return self.xc.get_setup_name()

    def calculate_paw_correction(self,
                                 setup,
                                 D_sp,
                                 dEdD_sp=None,
                                 addcoredensity=True,
                                 a=None):
        return self.xc.calculate_paw_correction(setup, D_sp, dEdD_sp,
                                                addcoredensity, a)

    def set_positions(self, spos_ac):
        self.ghat.set_positions(spos_ac)

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

        # SIC:
        self.esic = 0.0
        self.ekin = 0.0
        for spin in self.spin_s.values():
            if spin.kpt.psit_nG is not None:
                desic, dekin = spin.calculate()
                self.esic += desic
                self.ekin += dekin
        self.esic = self.kpt_comm.sum(self.esic)
        self.ekin = self.kpt_comm.sum(self.ekin)

        print ('esic from xc subroutine', self.esic)
        return exc + self.esic

    # def apply_orbital_dependent_hamiltonian(self,
    #                                         kpt,
    #                                         psit_nG,
    #                                         Htpsit_nG=None,
    #                                         dH_asp=None):
    #     spin = self.spin_s[kpt.s]
    #     if spin.W_mn is None:
    #         return
    #     spin.apply_orbital_dependent_hamiltonian(psit_nG)

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        spin = self.spin_s[kpt.s]
        if spin.W_mn is None:
            return
        spin.correct_hamiltonian_matrix(H_nn)

    def add_correction(self,
                       kpt,
                       psit_xG,
                       Htpsit_xG,
                       P_axi,
                       c_axi,
                       n_x,
                       calculate_change=False):
        spin = self.spin_s[kpt.s]
        if spin.W_mn is None:
            return

        if calculate_change:
            spin.calculate_residual_change(psit_xG, Htpsit_xG, P_axi, c_axi,
                                           n_x)
        else:
            spin.calculate_residual(psit_xG, Htpsit_xG, P_axi, c_axi)

    def rotate(self, kpt, U_nn):
        self.spin_s[kpt.s].rotate(U_nn)

    def setup_force_corrections(self, F_av):
        self.dF_av = np.zeros_like(F_av)
        for spin in self.spin_s.values():
            spin.add_forces(self.dF_av)
        self.wfs.kd.comm.sum(self.dF_av)

    def add_forces(self, F_av):
        F_av += self.dF_av

    def summary(self, log):
        for s in range(self.nspins):
            if s in self.spin_s:
                stabpot = self.spin_s[s].stabpot
                spin = self.spin_s[s]
                pos_mv = spin.get_centers()
                exc_m = spin.exc_m
                ecoulomb_m = spin.ecoulomb_m
                if self.kpt_comm.rank == 1 and self.finegd.comm.rank == 0:
                    nocc = self.kpt_comm.sum(spin.nocc)
                    self.kpt_comm.send(pos_mv, 0)
                    self.kpt_comm.send(exc_m, 0)
                    self.kpt_comm.send(ecoulomb_m, 0)
            else:
                if self.kpt_comm.rank == 0 and self.finegd.comm.rank == 0:
                    nocc = self.kpt_comm.sum(0)
                    pos_mv = np.zeros((nocc, 3))
                    exc_m = np.zeros(nocc)
                    ecoulomb_m = np.zeros(nocc)
                    self.kpt_comm.receive(pos_mv, 1)
                    self.kpt_comm.receive(exc_m, 1)
                    self.kpt_comm.receive(ecoulomb_m, 1)
            if self.kpt_comm.rank == 0 and self.finegd.comm.rank == 0:
                log('\nSIC orbital centers and energies:')
                log('                                %5.2fx   %5.2fx' %
                    (self.spin_s[0].xc_factor, self.spin_s[0].coulomb_factor))
                log('          x       y       z       XC    Coulomb')
                log('--------------------------------------------------')
                m = 0
                for pos_v, exc, ecoulomb in zip(pos_mv, exc_m, ecoulomb_m):
                    log('%3d  (%7.3f,%7.3f,%7.3f): %8.3f %8.3f' %
                        ((m, ) + tuple(pos_v) +
                         (exc * Hartree, ecoulomb * Hartree)))
                    m += 1
                log('--------------------------------------------------')
        log('\nTotal SIC energy     : %12.5f' % (self.esic * Hartree))
        log('Stabilizing potential: %12.5f' % (stabpot * Hartree))

    def read(self, reader):
        xc_factor = reader.hamiltonian.xc.sic_xc_factor
        coulomb_factor = reader.hamiltonian.xc.sic_coulomb_factor

        for s in range(self.nspins):
            W_mn = reader.hamiltonian.xc.get(
                'unitary_transformation{0}'.format(s))

            if s in self.spin_s:
                self.spin_s[s].initial_W_mn = W_mn
                self.spin_s[s].xc_factor = xc_factor
                self.spin_s[s].coulomb_factor = coulomb_factor

    def write(self, writer):
        for s in self.spin_s:
            spin = self.spin_s[s]
            writer.write(
                sic_xc_factor=spin.xc_factor,
                sic_coulomb_factor=spin.coulomb_factor)
            break

        for s in range(self.nspins):
            W_mn = self.get_unitary_transformation(s)

            if W_mn is not None:
                writer.write('unitary_transformation{0}'.format(s), W_mn)

    def get_unitary_transformation(self, s):
        if s in self.spin_s.keys():
            spin = self.spin_s[s]

            if spin.W_mn is None or spin.finegd.rank != 0:
                n = 0
            else:
                n = spin.W_mn.shape[0]
        else:
            n = 0

        n = self.wfs.world.sum(n)

        if n > 0:
            W_mn = np.zeros((n, n), dtype=self.dtype)
        else:
            W_mn = None
            return W_mn

        if s in self.spin_s.keys():
            spin = self.spin_s[s]
            #
            if spin.W_mn is None or spin.finegd.rank != 0:
                W_mn[:] = 0.0
            else:
                W_mn[:] = spin.W_mn[:]
            #
        else:
            W_mn[:] = 0.0

        self.wfs.world.sum(W_mn)
        return W_mn


class SICSpin:
    def __init__(self,
                 kpt,
                 xc,
                 density,
                 hamiltonian,
                 wfs,
                 poissonsolver,
                 ghat,
                 finegd,
                 coulomb_factor,
                 xc_factor,
                 dtype=complex,
                 uominres=1E-1,
                 uomaxres=1E-10,
                 uorelres=1E-4,
                 uonscres=1E-10,
                 rattle=0.0,
                 stabpot=0.0,
                 maxuoiter=10,
                 logging=2):
        """
        Developer: Ravindra Shinde
        Project  : FLOSIC for periodic systems

        A class for SIC objects per spin-polarization. This is used to compute the 
        self interaction energies using the Fermi Lowdin procedure for periodic systems.


        coulomb_factor:
            Scaling factor for Hartree-functional (0 to 1.0)

        xc_factor:
            Scaling factor for xc-functional (0 to 1.0)

        uominres:
            Minimum residual before unitary optimization starts

        uomaxres:
            Target accuracy for unitary optimization
            (absolute variance)

        uorelres:
            Target accuracy for unitary optimization
            (rel. to basis residual)

        maxuoiter:
            Maximum number of unitary optimization steps

        """
        self.wfs = wfs
        self.kpt = kpt
        self.xc = xc
        self.poissonsolver = poissonsolver
        self.ghat = ghat
        self.pt = wfs.pt
        self.density = density
        self.gd = wfs.gd
        self.finegd = finegd

        if self.finegd is self.gd:
            self.interpolator = None
            self.restrictor = None
        else:
            print ('finegd is not the same as gd  ', self.finegd is self.gd )            
            self.interpolator = Transformer(self.gd, self.finegd, 3)
            self.restrictor = Transformer(self.finegd, self.gd, 3)

        self.nspins = wfs.nspins
        self.spin = kpt.s
        self.timer = wfs.timer
        self.setups = wfs.setups
        self.nbands = wfs.bd.nbands        
        # print ('number of bands = ',self.nbands)

        self.dtype = dtype
        self.coulomb_factor = coulomb_factor
        self.xc_factor = xc_factor

        self.nocc = self.nbands     # number of occupied states
        self.W_mn = None            # unit. transf. to energy optimal states
        self.U_nn = np.identity(self.nocc)
        self.initial_W_mn = None  # initial unitary transformation
        self.vt_mG = None  # orbital dependent potentials

        self.vt_save_mG = self.gd.empty(1)  
#       orbital dependent potentials contribution from each orbital saved to the self.vt_save_mG
#       This will be used later to update the Hamiltonian

        self.exc_m = None       # SIC energy contribution (from E_xc)
        self.ecoulomb_m = None  # SIC energy contributions (from E_H)

        self.rattle = rattle  # perturb the initial unitary transformation
        self.stabpot = stabpot  # stabilizing constant potential to avoid
        # occupation of unoccupied states
        self.fermi_orbital = None
        self.uominres = uominres
        self.uomaxres = uomaxres
        self.uorelres = uorelres
        self.uonscres = uonscres
        self.maxuoiter = maxuoiter
        self.maxlsiter = 1  # maximum number of line-search steps
        self.maxcgiter = 2  # maximum number of CG-iterations
        self.lsinterp = True  # interpolate for minimum during line search
        self.basiserror = 1E+20
        self.logging = logging
        self.centers = 0
        for u, kpt in enumerate(wfs.kpt_u):      ## So far only Gamma point
            self.P_ani = kpt.P_ani

    def __repr__(self):
        return "<SICspin Class - SIC per polarization :: Functional = {0}, nspins = {1}>".format(self.xc, self.nspins)


    def initialize_orbitals(self, calc, rattle=0.0, localize=True):

        if not calc.wfs.gd.orthogonal:
            raise NotImplementedError('Wannier function analysis requires an orthogonal cell.')

        for kpoint in range(self.wfs.kd.nibzkpts):   # loop over k-points       

            if self.initial_W_mn is not None:
                self.nocc = self.initial_W_mn.shape[0]

            if self.nocc == 0:
                return

            Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
            for v in range(3):
                G_v = np.zeros(3)
                G_v[v] = 1
#                Z_mmv[:, :, v] = calc.get_wannier_integrals(self.spin, 0, 0, G_v, self.nocc)
                Z_mmv[:, :, v] = self.gd.wannier_matrix(
                    self.wfs.kpt_u[kpoint].psit_nG, self.wfs.kpt_u[kpoint].psit_nG, G_v, self.nocc)



            self.finegd.comm.sum(Z_mmv)

            if self.initial_W_mn is not None:
                self.W_mn = self.initial_W_mn

            elif localize:
                W_nm = np.identity(self.nocc)
                localization = 0.0
                for iter in range(100):
                    loc = _gpaw.localize(Z_mmv, W_nm)
                    if loc - localization < 1e-6:
                        break
                    localization = loc

                self.W_mn = W_nm.T.copy()
            else:
                self.W_mn = np.identity(self.nocc)

            if (rattle != 0.0 and self.W_mn is not None and
                    self.initial_W_mn is None):
                U_mm = random_unitary_matrix(rattle, self.nocc)
                self.W_mn = np.dot(U_mm, self.W_mn)

            if self.W_mn is not None:
                self.finegd.comm.broadcast(self.W_mn, 0)

            spos_mc = -np.angle(Z_mmv.diagonal()).T / (2 * pi)
            self.initial_pos_mv = np.dot(spos_mc % 1.0, self.gd.cell_cv) * Bohr


    def initialize_orbitals_copy(self, calc, rattle=0.0, localize=True):
        for kpoint in range(self.wfs.kd.nibzkpts):   # loop over k-points       

            if self.initial_W_mn is not None:
                self.nocc = self.initial_W_mn.shape[0]

            if self.nocc == 0:
                return

            Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
            for v in range(3):
                G_v = np.zeros(3)
                G_v[v] = 1
                Z_mmv[:, :, v] = self.gd.wannier_matrix(
                    self.wfs.kpt_u[kpoint].psit_nG, self.wfs.kpt_u[kpoint].psit_nG, G_v, self.nocc)
            self.finegd.comm.sum(Z_mmv)

            if self.initial_W_mn is not None:
                self.W_mn = self.initial_W_mn

            elif localize:
                W_nm = np.identity(self.nocc)
                localization = 0.0
                for iter in range(100):
                    loc = _gpaw.localize(Z_mmv, W_nm)
                    if loc - localization < 1e-6:
                        break
                    localization = loc
                    print ("localization in the sic module", localization)

                self.W_mn = W_nm.T.copy()
            else:
                self.W_mn = np.identity(self.nocc)

            if (rattle != 0.0 and self.W_mn is not None and
                    self.initial_W_mn is None):
                U_mm = random_unitary_matrix(rattle, self.nocc)
                self.W_mn = np.dot(U_mm, self.W_mn)

            if self.W_mn is not None:
                self.finegd.comm.broadcast(self.W_mn, 0)

            spos_mc = -np.angle(Z_mmv.diagonal()).T / (2 * pi)
            self.initial_pos_mv = np.dot(spos_mc % 1.0, self.gd.cell_cv)

    def localize_orbitals(self):

        assert self.gd.orthogonal

        # calculate wannier matrixelements
        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            G_v = np.zeros(3)
            G_v[v] = 1
            Z_mmv[:, :, v] = self.gd.wannier_matrix(
                self.kpt.psit_nG, self.kpt.psit_nG, G_v, self.nocc)
        self.finegd.comm.sum(Z_mmv)

        # setup the initial configuration (identity)
        W_nm = np.identity(self.nocc)

        # localize the orbitals
        localization = 0.0
        for iter in range(30):
            loc = _gpaw.localize(Z_mmv, W_nm)
            if loc - localization < 1e-6:
                break
            localization = loc

        # apply localizing transformation
        self.W_mn = W_nm.T.copy()

    def rattle_orbitals(self, rattle=-0.1):

        # check for the trivial cases
        if rattle == 0.0:
            return

        if self.W_mn is None:
            return

        # setup a "random" unitary matrix
        nocc = self.W_mn.shape[0]
        U_mm = random_unitary_matrix(rattle, nocc)

        # apply unitary transformation
        self.W_mn = np.dot(U_mm, self.W_mn)

    def get_centers(self):
        assert self.gd.orthogonal
        return self.initial_pos_mv


    def get_centers_copy(self):
        assert self.gd.orthogonal

        # calculate energy optimal states (if necessary)
        if self.phit_mG is None:
            self.update_optimal_states_kpts()

        # calculate wannier matrix elements
        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            G_v = np.zeros(3)
            G_v[v] = 1
            Z_mmv[:, :, v] = self.gd.wannier_matrix(self.phit_mG, self.phit_mG,
                                                    G_v, self.nocc)
        self.finegd.comm.sum(Z_mmv)

        # calculate positions of localized orbitals
        spos_mc = -np.angle(Z_mmv.diagonal()).T / (2 * pi)

        print ('printing the position centers')
        print(np.dot(spos_mc % 1.0, self.gd.cell_cv) * Bohr)
        return np.dot(spos_mc % 1.0, self.gd.cell_cv) * Bohr

#### This chain of functions is needed for obtaining the Lowdin Orbitals which will replace the wannier functions.

    def get_function(self, calc, u, n, pad=False):
        ''' To get the Kohn-Sham orbital at a given k-point and given band
        '''
        if pad:
            return calc.wfs.gd.zero_pad(calc.get_function(u , n, False))
        psit_nG = calc.wfs.kpt_u[u].psit_nG
        return psit_nG


    def get_wannier_function(self, u, n, pad=False):
        ''' To get the wannier function at a given k-point and given band. 
        Note that the wannier function still has a k-dependance. 
        '''
        if pad:
            return self.wfs.gd.zero_pad(self.get_wannier_function(u , n, False))
        psit_nG = self.wfs.kpt_u[u].psit_nG
        psit_nG = psit_nG.reshape((self.wfs.bd.nbands, -1))
        A = np.dot(self.U_nn[:, n], psit_nG).reshape(self.wfs.gd.n_c) / Bohr**1.5 
        return A


    def get_fermi_orbital(self, calc, kpoint, m, centers, pad=False):
        """ This function computes the Fermi Orbital for a given orbital index on the grid 
        based on the formula given in the manuscript. This function was saved as a cube file 
        and visualized and is found to be correct.
        """

        num_wann = np.shape(centers)[0]
        indices = np.zeros((3), dtype=int)
        wawc = np.zeros((num_wann), dtype=complex)
        square_wawc = np.zeros((num_wann), dtype=complex)
        fermi_orbital = np.zeros((calc.wfs.gd.N_c), dtype=complex)


        for n in range(num_wann):
            wann = self.get_wannier_function(kpoint, m, pad=False)
            indices_m = calc.wfs.gd.get_nearest_grid_point(centers[m]*Bohr, force_to_this_domain=True)
            numerator = np.conjugate(wann[tuple(indices_m)])*wann
            fermi_orbital  += numerator
            wawc[n] = np.abs(np.conjugate(wann[tuple(indices_m)])*wann[tuple(indices_m)])

        denominator = np.sqrt(np.sum(wawc))
        fermi_orbital = fermi_orbital/denominator

        return fermi_orbital


    def fermi_orbital_overlap_matrix(self, calc, kpoint, centers):
        """ This function computes the overlap between two Fermi orbitals. The size of the
        matrix is nband x nband. This matrix will be diagonalized later.
        """ 

        num_wann = np.shape(centers)[0]        
        block_comm = 0
        S_unn = np.zeros((num_wann, num_wann), dtype=complex)  
        overlap_matrix = np.zeros((num_wann, num_wann), dtype=complex)  

        P_ani = calc.wfs.kpt_u[kpoint].P_ani
        for iband in range(num_wann):
            for jband in range(num_wann):
                p0 = self.get_fermi_orbital(calc, kpoint, iband, centers)                
                p1 = self.get_fermi_orbital(calc, kpoint, jband, centers)
                S_unn[iband][jband] =  calc.wfs.integrate(p0,p1, global_integral=True)
                for a, P_ni in P_ani.items():
                    for i1 in range(len(P_ani[a][iband])):
                        for i2 in range(len(P_ni[jband])):
                            S_unn[iband][jband] += np.conj(P_ani[a][iband])[i1]*calc.wfs.setups[a].dO_ii[i1][i2]*P_ni[jband][i2]
        return S_unn


    def lowdin_orthogonalization(self, calc, kpoint, centers):
        """ This function gives the Lowdin orthogonalization transformation matrix by obtaining 
        the eigenvalues and eigenvectors of overlap matrix of fermi orbitals. """ 

        overlap_matrix = self.fermi_orbital_overlap_matrix(calc, kpoint, centers)

        # Following statement evaluates the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(overlap_matrix)


        if eigenvalues.any() <= 0.0 :
            lowdin_matrix = np.eye(len(eigenvalues))
            return lowdin_matrix

        eigenvalues = eigenvalues * np.eye(len(eigenvalues))

        eigenvalues_sqrt_inv = np.sqrt(np.linalg.inv(eigenvalues))
        lowdin_matrix = np.dot(eigenvectors, np.dot(eigenvalues_sqrt_inv, np.conjugate(eigenvectors.T)))

        return lowdin_matrix


    def orbital_transformation(self, kpoint, n, calc, centers):
        """ This function transforms all the original wannier functions by a unitary matrix based 
        on the Fermi Lowdin Procedure""" 

        num_wann = np.shape(centers)[0]
        unitary_matrix = self.lowdin_orthogonalization(calc, kpoint, centers)

        shape_all_orbs =  tuple(np.array([num_wann])) + tuple(calc.wfs.gd.n_c)
        all_old_orbitals = np.zeros(shape_all_orbs, dtype=complex)

        for i in range(num_wann):
            all_old_orbitals[i,:,:,:] = self.get_wannier_function(kpoint, i, pad=False)

        all_old_orbitals = all_old_orbitals.reshape((calc.wfs.bd.nbands, -1))
        new_orbitals = np.dot(unitary_matrix[:, n], all_old_orbitals).reshape(calc.wfs.gd.n_c)

        return new_orbitals



    def calculate_sic_matrixelements(self):
        
        # print ("printing vt_mG inside sic_matrixelement", self.vt_save_mG)
        # overlap of pseudo wavefunctions
        print ("printing shapes of Htphit_mG, phit_mG, vt_save_mG and Vmm, size of the grid")
        Htphit_mG = self.vt_save_mG * self.phit_mG
        # correct multiplication being done
        V_mm = np.zeros((self.nocc, self.nocc), dtype=self.dtype)
        gemm(self.gd.dv, self.phit_mG, Htphit_mG, 0.0, V_mm, 't')
        print (Htphit_mG.shape, self.phit_mG.shape, self.vt_save_mG.shape, V_mm.shape, self.wfs.gd.n_c)

        # PAW
        for a, P_mi in self.P_ami.items():
            for m, dH_p in enumerate(self.dH_amp[a]):
                dH_ii = unpack(dH_p)
                V_mm[m, :] += np.dot(P_mi[m], np.dot(dH_ii, P_mi.T))

        # accumulate over grid-domains
        self.finegd.comm.sum(V_mm)
        self.V_mm = V_mm

        # Symmetrization of V and kappa-matrix:
        K_mm = 0.5 * (V_mm - V_mm.T.conj())
#        V_mm = 0.5 * (V_mm + V_mm.T.conj())           # unhash later 

        ## Check the localization condition here itself. ##

        # # Update the Hamiltonian by adding the SIC correction to the original Hamiltonian
        self.ekin = -np.trace(V_mm) * (3 - self.nspins)

#        print ("the trace of V_mm", -np.trace(V_mm) )
#        print ("Printing the final SIC corrections", V_mm)

        return V_mm, K_mm, np.vdot(K_mm, K_mm).real

    def update_optimal_states_kpts(self, calc, centers):
#        print ("debug line inside updates states centers", centers)
        self.centers = centers
        for kpoint in range(self.wfs.kd.nibzkpts):   # loop over k-points               
            # pseudo wavefunctions
            self.phit_mG = self.wfs.gd.empty((self.wfs.kd.nibzkpts,self.nocc), self.wfs.dtype)

            for  i in range(self.nocc):
#                psit_nG = self.wfs.kpt_u[kpoint].psit_nG[i].reshape((self.wfs.bd.nbands, -1))
#                self.phit_mG[kpoint][i] = self.get_wannier_function(kpoint, i) #self.wfs.kpt_u[kpoint].psit_nG[i] #.reshape((self.wfs.bd.nbands, -1))         # dont forget to multiply 
#                self.phit_mG[kpoint][i] = np.dot(self.W_mn, psit_nG).reshape(calc.wfs.gd.n_c)
                self.phit_mG[kpoint][i] = self.orbital_transformation(kpoint, i, calc, centers)

            self.P_ami = {}
            for a, P_ni in self.wfs.kpt_u[kpoint].P_ani.items():
                self.P_ami[a] = np.dot(self.W_mn, P_ni[:self.nocc]) 


    def calculate_density(self, m, phit_G,  nt_sG):

        nt_sG[0].fill(0.0)
        for kpt in self.wfs.kpt_u:

            nt_G = nt_sG[kpt.s]            
            _gpaw.add_to_density(kpt.f_n[m], phit_G, nt_G)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.wfs.bd.comm.size == 1
                d_nn = np.zeros((self.wfs.bd.mynbands, self.wfs.bd.mynbands), dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for d_n, psi0_G in zip(d_nn, kpt.psit_nG): #kpt.psit_nG):
                    for d, psi_G in zip(d_n, kpt.psit_nG): #kpt.psit_nG):
                        if abs(d) > 1.e-12:
                            nt_G += (psi0_G.conj() * d * psi_G).real


        self.wfs.kptband_comm.sum(nt_sG)


        # PAW corrections
        Q_aL = {}
        D_ap = {}
        for a, P_mi in self.P_ami.items():
            P_i = P_mi[0]
            D_ii = np.outer(P_i, P_i.conj()).real
            D_ap[a] = D_p = pack(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        self.timer.start('Symmetrize density in the self-interaction routine')
        for nt_G in nt_sG:
            self.wfs.kd.symmetry.symmetrize(nt_G, self.gd)
        self.timer.stop('Symmetrize density in the self-interaction routine')

        return nt_sG[0], Q_aL, D_ap


#    def update_potentials(self, calc, save=False, restore=False):
    def update_potentials(self, calc, save=False, restore=False):        
        if restore:
            self.exc_m = self.exc_save_m
            self.ecoulomb_m = self.eco_save_m
            self.esic = self.esic_save
            self.vt_mG = self.vt_save_mG.copy()
            self.dH_amp = self.dH_save_amp.copy()
            return

        self.timer.start('update densities and potentials for SIC')

#        print ("debug from sic object: printing wannier centers", self.get_centers())
        # nt_sg = self.gd.empty(2)
        # nt_sg[1] = 0.0
        # vt_sg = self.gd.empty(2)

        # PAW
        W_aL = self.ghat.dict()
        zero_initial_phi = False

#        initialize some bigger fields
        if self.vt_mG is None : #or self.nocc != self.phit_mG.shape[0]:
            self.vt_mG = self.gd.empty(self.nocc)
            self.exc_m = np.zeros(self.nocc)
            self.ecoulomb_m = np.zeros(self.nocc)
#            self.vHt_mg = self.finegd.zeros(self.nocc)
            self.vHt_mg = self.gd.zeros(self.nocc)            
            zero_initial_phi = True

        #
        # PAW
        self.dH_amp = {}
        for a, P_ni in self.P_ani.items():
            ni = P_ni.shape[1]
            self.dH_amp[a] = np.empty((self.nocc, ni * (ni + 1) // 2))
#        print ('self.dH_amp', self.dH_amp)
        #
        self.Q_maL = {}
        # loop all occupied wannier orbitals
#        for kpoint in range(self.wfs.kd.nibzkpts):
#################################################################################### comment three lines below
        wfl_density = self.wfs.gd.empty(self.nspins) 
#   These two lines are for debugging        
        wfl_density_check = self.wfs.gd.empty(self.nspins)         
        wfl_density_check.fill(0.0)
#       added portion of code : remove later
        wfl_density.fill(0.0)
        for kpt in self.wfs.kpt_u:
            self.wfs.add_to_density_from_k_point(wfl_density, kpt)
        self.wfs.kptband_comm.sum(wfl_density)

        self.wfs.timer.start('Ravindra Symmetrize density')
        for nt_G in wfl_density:
            self.wfs.kd.symmetry.symmetrize(nt_G, self.wfs.gd)
        self.wfs.timer.stop('Ravindra Symmetrize density')
        
        wfl_density[:] += self.density.nct_G

        # for m, phit_G in zip(range(self.nbands), self.kpt.psit_nG): #enumerate(self.phit_mG[:][:]):
        #     nt_sg[0], Q_aL, D_ap = self.calculate_density(phit_G, nt_sg, m)
        #     wfl_density += nt_sg[0]
#                vt_sg[:] = 0.0

### Following portions uncommented for review. get the orbital density only from the above snippets. Jan 03 12:00PM
## Outside the k-loop

#        wfl_density = self.wfs.gd.empty(self.nspins)  ### Unhash later lines 983 to 1071
        # for m, phit_G in enumerate(self.phit_mG):                
        
        for m in range(self.nocc):
            # xc-SIC
            nt_sg = self.gd.empty(2)
            nt_sg[1] = 0.0
            vt_sg = self.gd.empty(2)
#       Check the alternative approach below. Explicit loop on kpoints
#            local_phit_mG = self.phit_mG[:,m,:,:,:]
            # print ("whole phit_mG shape", self.phit_mG.shape)            
            # print ("debug shape", local_phit_mG.shape)
            # setup the single-particle density and PAW density-matrix
#           just for checking . delete after debug            
#            print ("printing the first wavefunction at gamma in sic code", calc.wfs.kpt_u[0].psit_nG[m][0])
#            nt_sg[0], Q_aL, D_ap = self.calculate_density(m, self.wfs.kpt_u[0].psit_nG[m], nt_sg)
            nt_sg[0], Q_aL, D_ap = self.calculate_density(m, self.phit_mG[:,m,:,:,:], nt_sg)
            vt_sg[:] = 0.0
            wfl_density_check +=  nt_sg[0]   # Ravindra

            self.timer.start('XC')
            if self.xc_factor != 0.0:
#                exc = self.xc.calculate(self.gd, nt_sg, vt_sg)                
#                print ("the comparison of exc with corrected densities", exc)
                exc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
#                exc /= self.gd.comm.size
                exc /= self.finegd.comm.size                
                vt_sg[0] *= -self.xc_factor
                #
                # PAW Correction part. Keep it untouched
                for a, D_p in D_ap.items():
                    setup = self.setups[a]
                    dH_p = self.dH_amp[a][m]
                    dH_sp = np.zeros((2, len(dH_p)))
                    #
                    D_sp = np.array([D_p, np.zeros_like(D_p)])
                    exc += self.xc.calculate_paw_correction(
                        setup, D_sp, dH_sp, addcoredensity=False)
                    dH_p[:] = -dH_sp[0] * self.xc_factor

                self.exc_m[m] = -self.xc_factor * exc
                # print ('exc in loop', self.exc_m[m])
            self.timer.stop('XC')
          

            #
            #Hartree-SIC
            self.timer.start('Hartree')
            if self.coulomb_factor != 0.0:
                #
                # add compensation charges to pseudo density
                #self.ghat.add(nt_sg[0], Q_aL)
                self.poissonsolver.solve(
                    self.vHt_mg[m],
                    nt_sg[0],
                    zero_initial_phi=zero_initial_phi)
                ecoulomb = 0.5 * self.finegd.integrate(nt_sg[0] *
                                                       self.vHt_mg[m], global_integral=False)
                #ecoulomb /= self.gd.comm.size
                # print ('ecoulomb in loop', ecoulomb)
                # ecoulomb = 0.5 * self.finegd.integrate(nt_sg[0] *
                #                                        self.vHt_mg[m])
                ecoulomb /= self.finegd.comm.size

                vt_sg[0] -= self.coulomb_factor * self.vHt_mg[m]
                #
                # PAW
                self.ghat.integrate(self.vHt_mg[m], W_aL)
                for a, D_p in D_ap.items():
                    setup = self.setups[a]
                    dH_p = self.dH_amp[a][m]
                    M_p = np.dot(setup.M_pp, D_p)
                    ecoulomb += np.dot(D_p, M_p)
                    #
                    dH_p -= self.coulomb_factor * (
                        2.0 * M_p + np.dot(setup.Delta_pL, W_aL[a]))
                #
                self.ecoulomb_m[m] = -self.coulomb_factor * ecoulomb
                self.timer.stop('Hartree')
          
            self.vt_mG[m] = vt_sg[0]           
            self.vt_save_mG += self.vt_mG[m]
#            print ("printing header of vt_save_mG each orbital", self.vt_save_mG[0][0][:] )
            self.Q_maL[m] = Q_aL

        self.timer.stop('update densities and potentials for SIC')

#         # accumulate total xc-SIC and coulomb-SIC
        self.finegd.comm.sum(self.exc_m)
        self.finegd.comm.sum(self.ecoulomb_m)

#         # total sic (including spin-degeneracy)
        self.esic = self.exc_m.sum()
        self.esic += self.ecoulomb_m.sum()
#        nt_sG[0], Q_aL, D_ap = calculate_density(self, m, phit_G,  nt_sG)
        wfl_density_check += self.density.nct_G
        
#       Do the density update here.
        print ("Printing shapes of original densities and wfl densities")
        print ("nt_sG shape", calc.density.nt_sG.shape)
        print ("wfl_density shape", wfl_density.shape)        
#        calc.density.nt_sG[0,:,:,:] = wfl_density[0,:,:,:]
        calc.density.nt_sG[0,:,:,:] = wfl_density_check/(self.gd.integrate(wfl_density_check))

## write update wavefunction object here
        print ("wann centers before updating wavefunctions", self.centers)
        self.update_sic_wavefunctions(calc)

        calc.write_wannier_gpw('wannier.gpw', wfl_density, mode='all')        
        calc.write_wannier_gpw('wannier_check.gpw', calc.density.nt_sG, mode='all')        


        print ("integrating densities from FLOSIC", self.gd.integrate(calc.density.nt_sG) )        
#        print ("integrating densities from Wannier module", self.gd.integrate(wfl_density) )
#        print ("integrating densities from sic module", self.gd.integrate(wfl_density_check) )        

        print ('ESIC at the end per spin component', self.esic)                                
        self.esic *= (3 - self.nspins)

        V_mm, K_mm, Real_Vdot = self.calculate_sic_matrixelements()


        ham = calc.hamiltonian

        ham.update_hamiltonian_sic(calc.density, wfl_density_check, self.ecoulomb_m.sum(), self.exc_m.sum() )
        ham.get_energy_sic(calc.occupations, self.esic)

#        print ("V_mm", np.real(V_mm))
#        print ("K_mm", K_mm)
#        print ("Real_Vdot", Real_Vdot)

        # if save:
        #     self.exc_save_m = self.exc_m
        #     self.eco_save_m = self.ecoulomb_m
        #     self.esic_save = self.esic
        #     self.vt_save_mG = self.vt_mG.copy()
        #     self.dH_save_amp = self.dH_amp.copy()


    def update_sic_wavefunctions(self, calc):

        for kpoint in range(self.wfs.kd.nibzkpts):   # loop over k-points               
            for  i in range(self.nocc):
                self.wfs.kpt_u[kpoint].psit_nG[i] = self.phit_mG[kpoint][i]

            self.P_ami = {}
            for a, P_ni in self.wfs.kpt_u[kpoint].P_ani.items():
                self.P_ami[a] = np.dot(self.W_mn, P_ni[:self.nocc]) 



    def apply_orbital_dependent_hamiltonian(self, psit_nG):
        """...

        Setup::

            |V phi_m> and <l|Vphi_m>,

        for occupied states m and unoccupied states l."""

        # nocc = self.nocc
        # nvirt = psit_nG.shape[0] - nocc

        self.Htphit_mG = self.vt_mG * self.phit_mG

    def correct_hamiltonian_matrix(self, H_nn):
        """ Add contributions of the non-local part of the
            interaction potential to the Hamiltonian matrix.

            on entry:
                          H_nn[n,m] = <n|H_{KS}|m>
            on exit:
                          H_nn[n,m] = <n|H_{KS} + V_{u}|m>

            where V_u is the unified Hamiltonian

                V_u = ...

        """

        nocc = self.nocc
        nvirt = H_nn.shape[0] - nocc

        W_mn = self.W_mn
        # R_mk = self.R_mk

        if self.gd.comm.rank == 0:
            V_mm = 0.5 * (self.V_mm + self.V_mm.T)
            H_nn[:nocc, :nocc] += np.dot(W_mn.T, np.dot(V_mm, W_mn))
            if self.stabpot != 0.0:
                H_nn[nocc:, nocc:] += np.eye(nvirt) * self.stabpot

        if nvirt != 0:
            H_nn[:nocc, nocc:] = 0.0  # R_nk
            H_nn[nocc:, :nocc] = 0.0  # R_nk.T
            # R_nk = np.dot(W_mn.T, R_mk) # CHECK THIS
            # H_nn[:nocc, nocc:] += R_nk
            # H_nn[nocc:, :nocc] += R_nk.T

    def calculate_residual(self, psit_nG, Htpsit_nG, P_ani, c_ani):
        """ Calculate the action of the unified Hamiltonian on an
            arbitrary state:

                H_u|Psi> =
        """

        nocc = self.nocc
        nvirt = psit_nG.shape[0] - nocc

        # constraint for unoccupied states
        R_mk = np.zeros((nocc, nvirt), dtype=self.dtype)
        if nvirt > 0:
            gemm(self.gd.dv, psit_nG[nocc:], self.Htphit_mG, 0.0, R_mk, 't')
            # PAW
            for a, P_mi in self.P_ami.items():
                P_ni = P_ani[a]

                for m, dH_p in enumerate(self.dH_amp[a]):
                    dH_ii = unpack(dH_p)
                    R_mk[m] += np.dot(P_mi[m], np.dot(dH_ii, P_ni[nocc:].T))

            self.finegd.comm.sum(R_mk)

        # self.R_mk = R_mk
        # R_mk  = self.R_mk
        W_mn = self.W_mn
        Htphit_mG = self.Htphit_mG
        phit_mG = self.phit_mG
        K_mm = 0.5 * (self.V_mm - self.V_mm.T)
        Q_mn = np.dot(K_mm, W_mn)

        # Action of unified Hamiltonian on occupied states:
        if nocc > 0:
            gemm(1.0, Htphit_mG, W_mn.T.copy(), 1.0, Htpsit_nG[:nocc])
            gemm(1.0, phit_mG, Q_mn.T.copy(), 1.0, Htpsit_nG[:nocc])
        if nvirt > 0:
            gemm(1.0, phit_mG, R_mk.T.copy(), 1.0, Htpsit_nG[nocc:])
            if self.stabpot != 0.0:
                Htpsit_nG[nocc:] += self.stabpot * psit_nG[nocc:]

        # PAW
        for a, P_mi in self.P_ami.items():
            #
            c_ni = c_ani[a]
            ct_mi = P_mi.copy()
            #
            dO_ii = self.setups[a].dO_ii
            c_ni[:nocc] += np.dot(Q_mn.T, np.dot(P_mi, dO_ii))
            c_ni[nocc:] += np.dot(R_mk.T, np.dot(P_mi, dO_ii))
            #
            for m, dH_p in enumerate(self.dH_amp[a]):
                dH_ii = unpack(dH_p)
                ct_mi[m] = np.dot(P_mi[m], dH_ii)
            c_ni[:nocc] += np.dot(W_mn.T, ct_mi)
            c_ni[nocc:] += self.stabpot * np.dot(P_ani[a][nocc:], dO_ii)

    def calculate_residual_change(self, psit_xG, Htpsit_xG, P_axi, c_axi, n_x):
        """

        """
        assert len(n_x) == 1

        Htphit_mG = self.Htphit_mG
        phit_mG = self.phit_mG

        w_mx = np.zeros((self.nocc, 1), dtype=self.dtype)
        v_mx = np.zeros((self.nocc, 1), dtype=self.dtype)

        gemm(self.gd.dv, psit_xG, phit_mG, 0.0, w_mx, 't')
        gemm(self.gd.dv, psit_xG, Htphit_mG, 0.0, v_mx, 't')

        # PAW
        for a, P_mi in self.P_ami.items():
            P_xi = P_axi[a]
            dO_ii = self.setups[a].dO_ii
            #
            w_mx += np.dot(P_mi, np.dot(dO_ii, P_xi.T))

            for m, dH_p in enumerate(self.dH_amp[a]):
                dH_ii = unpack(dH_p)
                v_mx[m] += np.dot(P_mi[m], np.dot(dH_ii, P_xi.T))

        # sum over grid-domains
        self.finegd.comm.sum(w_mx)
        self.finegd.comm.sum(v_mx)

        V_mm = 0.5 * (self.V_mm + self.V_mm.T)
        q_mx = v_mx - np.dot(V_mm, w_mx)

        if self.stabpot != 0.0:
            q_mx -= self.stabpot * w_mx

        gemm(1.0, Htphit_mG, w_mx.T.copy(), 1.0, Htpsit_xG)
        gemm(1.0, phit_mG, q_mx.T.copy(), 1.0, Htpsit_xG)

        # PAW
        for a, P_mi in self.P_ami.items():
            c_xi = c_axi[a]
            ct_mi = P_mi.copy()

            dO_ii = self.setups[a].dO_ii
            c_xi += np.dot(q_mx.T, np.dot(P_mi, dO_ii))

            for m, dH_p in enumerate(self.dH_amp[a]):
                dH_ii = unpack(dH_p)
                ct_mi[m] = np.dot(P_mi[m], dH_ii)
            c_xi += np.dot(w_mx.T, ct_mi)

        if self.stabpot != 0.0:
            Htphit_mG += self.stabpot * psit_xG
            for a, P_xi in P_axi.items():
                dO_ii = self.setups[a].dO_ii
                c_axi[a] += self.stabpot * np.dot(P_xi, dO_ii)

    def rotate(self, U_nn):
        """ Unitary transformations amongst the canonic states
            require to apply a counter-acting transformation to
            the energy optimal states. This subroutine takes
            care of it.

            Reorthogonalization is required whenever unoccupied
            states are mixed in.
        """
        # skip if no transformation to optimal states is set-up
        if self.W_mn is None:
            return

        # compensate the transformation amongst the occupied states
        self.W_mn = np.dot(self.W_mn, U_nn[:self.nocc, :self.nocc])

        # reorthogonalize if unoccupied states may have been mixed in
        if self.nocc != U_nn.shape[0]:
            self.W_mn = ortho(self.W_mn)
            # self.R_mk = np.dot(self.R_mk, U_nn[self.nocc:, self.nocc:].T)

    def add_forces(self, F_av):
        # Calculate changes in projections
        deg = 3 - self.nspins
        F_amiv = self.pt.dict(self.nocc, derivative=True)
        self.pt.derivative(self.phit_mG, F_amiv)
        for m in range(self.nocc):
            # Force from compensation charges:
            dF_aLv = self.ghat.dict(derivative=True)
            self.ghat.derivative(self.vHt_mg[m], dF_aLv)
            for a, dF_Lv in dF_aLv.items():
                F_av[a] -= deg * self.coulomb_factor * \
                    np.dot(self.Q_maL[m][a], dF_Lv)

            # Force from projectors
            for a, F_miv in F_amiv.items():
                F_vi = F_miv[m].T.conj()
                dH_ii = unpack(self.dH_amp[a][m])
                P_i = self.P_ami[a][m]
                F_v = np.dot(np.dot(F_vi, dH_ii), P_i)
                F_av[a] += deg * 2 * F_v.real

    def calculate(self):
        """ Evaluate the non-unitary invariant part of the
            energy functional and returns

            esic: float
                self-interaction energy

            ekin: float
                correction to the kinetic energy
        """
        # initialize transformation from canonic to energy
        # optimal states (if necessary)
        if self.W_mn is None:
            self.initialize_orbitals(rattle=self.rattle)

        # optimize the non-unitary invariant part of the
        # functional
        self.unitary_optimization()

        return self.esic, self.ekin

    def unitary_optimization(self):
        """ Optimization of the non-unitary invariant part of the
            energy functional.
        """

        optstep = 0.0
        Gold = 0.0
        cgiter = 0
        #
        epsstep = 0.005  # 0.005
        dltstep = 0.1  # 0.1
        prec = 1E-7

        # get the initial ODD potentials/energy/matrixelements
        self.update_optimal_states()
        self.update_potentials(save=True)
        ESI = self.esic
        V_mm, K_mm, norm = self.calculate_sic_matrixelements()

        if norm < self.uonscres and self.maxuoiter > 0:
            return

        if self.nocc <= 1:
            return

        # optimize the unitary transformation
        #
        # allocate arrays for the search direction,
        # i.e., the (conjugate) gradient
        D_old_mm = np.zeros_like(self.W_mn)

        for iter in range(abs(self.maxuoiter)):
            # copy the initial unitary transformation and orbital
            # dependent energies
            W_old_mn = self.W_mn.copy()

            # setup the steepest-descent/conjugate gradient
            # D_nn:  search direction
            # K_nn:  inverse gradient
            # G0  :  <K,D> (projected length of D along K)
            if Gold != 0.0:
                # conjugate gradient
                G0 = np.sum(K_mm * K_mm.conj()).real
                beta = G0 / Gold
                Gold = G0
                D_mm = K_mm + beta * D_old_mm

                G0 = np.sum(K_mm * D_mm.conj()).real
            else:
                # steepest-descent
                G0 = np.sum(K_mm * K_mm.conj()).real
                Gold = G0
                D_mm = K_mm

            updated = False
            minimum = False
            failed = True
            noise = False
            E0 = ESI

            # try to estimate optimal step-length from change in length
            # of the gradient (force-only)
            if (epsstep != 0.0 and norm > self.uomaxres):
                step = epsstep
                while (True):
                    U_mm = matrix_exponential(D_mm, step)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    E1 = self.esic
                    K0_mm = K_mm.copy()
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()

                    # projected length of the gradient at the new position
                    G1 = np.sum(((K_mm - K0_mm) / step) * D_mm.conj()).real
                    #
                    if (abs(E1 - E0) < prec and E1 >= E0):
                        #
                        eps_works = True
                        Eeps = E1
                        noise = True
                    elif (E1 < E0):
                        #
                        # trial step reduced energy
                        eps_works = True
                        Eeps = E1
                    else:
                        #
                        # epsilon step did not work
                        eps_works = False
                        optstep = 0.0
                        break

                    # compute the optimal step size
                    # optstep = step*G0/(G0-G1)
                    # print -G0/G1
                    optstep = -G0 / G1

                    if (eps_works):
                        break

                    # print eps_works, optstep, G0/((G0-G1)/step)

                    # decide on the method for stepping
                if (optstep > 0.0):

                    # convex region -> force only estimate for minimum
                    U_mm = matrix_exponential(D_mm, optstep)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    E1 = self.esic
                    if (abs(E1 - E0) < prec and E1 >= E0):
                        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                        ESI = E1
                        optstep = optstep
                        lsiter = 0
                        maxlsiter = -1
                        updated = True
                        minimum = True
                        failed = False
                        lsmethod = 'CV-N'
                        noise = True
                    elif (E1 < E0):
                        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                        ESI = E1
                        optstep = optstep
                        lsiter = 0
                        maxlsiter = -1
                        updated = True
                        minimum = True
                        failed = False
                        lsmethod = 'CV-S'
                    else:
                        # self.K_unn[q] = K_nn
                        ESI = E0
                        step = optstep
                        optstep = 0.0
                        lsiter = 0
                        maxlsiter = self.maxlsiter
                        updated = False
                        minimum = False
                        failed = True
                        lsmethod = 'CV-F-CC'
                else:
                    # self.K_unn[q] = K_nn
                    ESI = E0
                    step = optstep
                    optstep = 0.0
                    lsiter = 0
                    maxlsiter = self.maxlsiter
                    updated = False
                    minimum = False
                    failed = True
                    lsmethod = 'CC'
            else:
                maxlsiter = 0
                lsiter = -1
                optstep = epsstep
                updated = False
                minimum = True
                failed = False
                lsmethod = 'CC-EPS'

            if (optstep == 0.0):
                #
                # we are in the concave region or force-only estimate failed,
                # just follow the (conjugate) gradient
                step = dltstep * abs(step)
                U_mm = matrix_exponential(D_mm, step)
                self.W_mn = np.dot(U_mm, W_old_mn)
                self.update_optimal_states()
                self.update_potentials()
                E1 = self.esic
                #
                #
                if (abs(E1 - E0) < prec and E1 >= E0):
                    ESI = E1
                    optstep = 0.0
                    updated = False
                    minimum = True
                    failed = True
                    lsmethod = lsmethod + '-DLT-N'
                    noise = True
                    maxlsiter = -1
                elif (E1 < E0):
                    ESI = E1
                    optstep = step
                    updated = True
                    minimum = False
                    failed = False
                    lsmethod = lsmethod + '-DLT'
                    maxlsiter = self.maxlsiter
                elif (eps_works):
                    ESI = Eeps
                    E1 = Eeps
                    step = epsstep
                    updated = False
                    minimum = False
                    failed = False
                    lsmethod = lsmethod + '-EPS'
                    maxlsiter = self.maxlsiter
                else:
                    optstep = 0.0
                    updated = False
                    minimum = False
                    failed = True
                    lsmethod = lsmethod + '-EPS-F'
                    maxlsiter = -1
                #
                G = (E1 - E0) / step
                step0 = 0.0
                step1 = step
                step2 = 2 * step
                #
                for lsiter in range(maxlsiter):
                    #
                    # energy at the new position
                    U_mm = matrix_exponential(D_mm, step2)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    E2 = self.esic
                    G = (E2 - E1) / (step2 - step1)
                    #
                    #
                    if (G > 0.0):
                        if self.lsinterp:
                            a = (E0 / ((step2 - step0) * (step1 - step0)) +
                                 E2 / ((step2 - step1) * (step2 - step0)) -
                                 E1 / ((step2 - step1) * (step1 - step0)))
                            b = (E2 - E0) / (step2 - step0) - a * (step2 +
                                                                   step0)
                            if (a < step**2):
                                optstep = 0.5 * (step0 + step2)
                            else:
                                optstep = -0.5 * b / a
                            updated = False
                            minimum = True
                            break
                        else:
                            optstep = step1
                            updated = False
                            minimum = True
                            break

                    elif (G < 0.0):
                        optstep = step2
                        step0 = step1
                        step1 = step2
                        step2 = step2 + step
                        E0 = E1
                        E1 = E2
                        ESI = E2
                        updated = True
                        minimum = False

            if (cgiter != 0):
                lsmethod = lsmethod + ' CG'

            if (cgiter >= self.maxcgiter or not minimum):
                Gold = 0.0
                cgiter = 0
            else:
                cgiter = cgiter + 1
                D_old_mm[:, :] = D_mm[:, :]

            # update the energy and matrixelements of V and Kappa
            # and accumulate total residual of unitary optimization
            if (not updated):
                if optstep > 0.0:
                    U_mm = matrix_exponential(D_mm, optstep)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    self.update_potentials()
                    ESI = self.esic
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                else:
                    self.W_mn = W_old_mn
                    self.update_optimal_states()
                    self.update_potentials(restore=True)
                    ESI = self.esic
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()

            if (lsiter == maxlsiter - 1):
                V_mm, K_mm, norm = self.calculate_sic_matrixelements()

            E0 = ESI

            # orthonormalize the energy optimal orbitals
            self.W_mn = ortho(self.W_mn)
            K = max(norm, 1.0e-16)

            if self.finegd.comm.rank == 0:
                if self.logging == 1:
                    print("           UO-%d: %2d %5.1f  %20.5f  " %
                          (self.spin, iter, np.log10(K), ESI * Hartree))
                elif self.logging == 2:
                    print("           UO-%d: %2d %5.1f  %20.5f  " %
                          (self.spin, iter, np.log10(K), ESI * Hartree) +
                          lsmethod)

            if ((K < self.uomaxres or
                 K < self.wfs.eigensolver.error * self.uorelres) or noise or
                    failed) and not self.maxuoiter < 0:
                break
