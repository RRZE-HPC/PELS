#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import numpy as np
from scipy.sparse import *
import re

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




def create_matrix(matstring, imbal=False):

    label, nx, ny, nz = parse_matstring(matstring)
    if label == 'Laplace' and nz == 1:
        A=create_laplacian(nx,ny)
    elif label == 'Laplace':
        A=create_laplacian3d(nx,ny,nz)
    else:
        raise(ValueError('create_matrix can only build "Laplace<nx>x<ny>", "Laplace<nx>x<ny>x<nz>",  matrices up to now.'))
    
    if(imbal==False):
        return A
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

        return A+arrow

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

