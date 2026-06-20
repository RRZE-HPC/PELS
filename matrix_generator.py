#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import sys
import numpy as np
import scipy
from scipy.sparse import *
from scipy.io import mmread
import re
from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm

try:
    import pyamg
except ImportError:
    pyamg = None

def parse_matstring(input):
    '''
    From an input like  'Laplace128x256',
    extract name, nx and ny
    '''
    # The string should start with some non-digit characters (\D+) that form the label,
    # followed by <nx>x<ny>, where nx and ny are integers with an arbitrary number of digits  (\d+)
    input_list = re.match(r"(?P<label>[-+]?\D+)(?P<nx>[-+]?\d+)x(?P<ny>[-+]?\d+)", input)
    input_list3d = re.match(r"(?P<label>[-+]?\D+)(?P<nx>[-+]?\d+)x(?P<ny>[-+]?\d+)x(?P<nz>[-+]?\d+)", input)
    if input_list == None and input_list3d == None:
        raise(ValueError('Could not parse matrix genration string, should have the format "<label><nx>x<ny>x<nz>", where <label> is a string, nx, ny and nz (in case of 3d) are integers.'))

    if input_list3d != None:
        label = input_list3d['label']
        nx = int(input_list3d['nx'])
        ny = int(input_list3d['ny'])
        nz = int(input_list3d['nz'])
    elif input_list != None:
        label = input_list['label']
        nx = int(input_list['nx'])
        ny = int(input_list['ny'])
        nz = 1

    return label, nx, ny, nz


def rcm_reordering(A):
    p = rcm(A)
    return A[p, :][:, p], p






def create_matrix(matstring, rcm=False, imbal=False):

    if matstring.endswith(('.mm','.mtx','.mm.gz','.mtx.gz')):
        A = csr_matrix(mmread(matstring))
    else:
        label, nx, ny, nz = parse_matstring(matstring)
        if label == 'Laplace' and nz == 1:
            A = create_laplacian(nx,ny)
        elif label == 'Laplace':
            A=create_laplacian3d(nx,ny,nz)
        elif pyamg is not None:
            if label == 'LinElast':
                if nz>1:
                    raise(ValueError('LinElast<nx>x<ny> only available in 2D'))
                # linear elasticity stiffness matrix A (on a regular, quadrilateral mesh)
                # B are the rigid-body modes (the nullspace of A), we ignore them but they
                # could be used for preconditioning.
                A, B = pyamg.gallery.linear_elasticity((nx, ny), format='csr')
                A = csr_matrix(A)
                # Could also use a triangular mesh:
                #E2V,Vert = regular_triangle_mesh(nx, ny)
                #A, B = pyamg.gallery.linear_elasticity_p1(Vert, E2V, format='csr')
            else:
                raise(ValueError('matrix string "'+label+'" not supported, only "Laplace" or "LinElast" are, right now,\n'
                                 'or read matrix market (*.mm[.gz] or *.mtx[.gz])'))

    p = None
    if rcm:
        A, p = rcm_reordering(A)

    if(imbal==False):
        return A, p
    else:
        N=A.shape[0]
        rowInd=np.zeros(2*N-1, dtype=int)
        colInd=np.zeros(2*N-1, dtype=int)
        data=-1*np.ones(2*N-1)
        #create botom row
        for i in range(0,N):
            rowInd[i]=N-1
            colInd[i]=i
        #create right row
        for i in range(0,N-1):
            colInd[N+i]=N-1
            rowInd[N+i]=i
        arrow=csr_matrix((data, (rowInd, colInd)), shape=(N,N))

        return A+arrow, p

def create_laplacian(nx,ny):
    N=nx*ny
    ex=np.ones([nx])
    ey=np.ones([ny])
    Ix=eye(nx)
    Iy=eye(ny)
    Dx=spdiags([-ex,2*ex,-ex],[-1,0,1],nx,nx)
    Dy=spdiags([-ey,2*ey,-ey],[-1,0,1],ny,ny)
    A=kron(Dx,Iy) + kron(Ix,Dy)
    return A

def create_laplacian3d(nx,ny,nz):
    N=nx*ny*nz
    ex=np.ones([nx])
    ey=np.ones([ny])
    ez=np.ones([nz])
    Ix=eye(nx)
    Iy=eye(ny)
    Iz=eye(nz)
    Dx=spdiags([-ex,2*ex,-ex],[-1,0,1],nx,nx)
    Dy=spdiags([-ey,2*ey,-ey],[-1,0,1],ny,ny)
    Dz=spdiags([-ez,2*ez,-ez],[-1,0,1],nz,nz)
    A=kron(kron(Dx,Iy), Iz) + kron(kron(Ix,Dy), Iz) + kron(Ix, kron(Iy,Dz))
    return A

if __name__=='__main__':
    # generate a matrix from the given string and store it in a MatrixMarket file
    # Example: python matrix_generator.py LinElast20x20 -> LinElast20x20.mtx
    if len(sys.argv)>1:
        matstr = sys.argv[1]
        A=create_matrix(matstr)
        scipy.io.mmwrite(matstr, A, 'Generated by PELS/matrix_generator.create_matrix()')
