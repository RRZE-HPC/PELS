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
    if input_list == None:
        raise(ValueError('Could not parse matrix genration string, should have the format "<label><nx>x<ny>", where <label> is a string, nx and ny are integers.'))
    label = input_list['label']
    nx = int(input_list['nx'])
    ny = int(input_list['ny'])
    return label, nx, ny



def create_matrix(matstring):

    label, nx, ny = parse_matstring(matstring)
    if label == 'Laplace':
        return create_laplacian(nx,ny)
    else:
        raise(ValueError('create_matrix can only build "Laplace<nx>x<ny>" matrices up to now.'))

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
