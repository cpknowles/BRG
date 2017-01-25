from pylab import *
import subprocess
import cPickle as pickle

from scipy.ndimage.filters import convolve

# Load precomputed outline
outline_0 = array(pickle.load(open('outline.p'))).T
outline_0 = hstack((outline_0,ones((outline_0.shape[0],1))))

# Skip boundary points (necessary so that each small edge segment isn't meshed)
skip = 29
outline_0 = outline_0[::skip,:]

# Nominal cellsize
cellsize = 200

f = open('outline.geo','w')
f.write('Mesh.CharacteristicLengthMin={0};\n'.format(cellsize))
f.write('Mesh.CharacteristicLengthMax={0};\n'.format(cellsize))
f.write('Mesh.CharacteristicLengthExtendFromBoundary=0;\n')
f.write('lc={0};\n'.format(cellsize))

surface_indicators = unique(outline_0[:,-1]).astype(int)

for s in surface_indicators:
    outline = outline_0[outline_0[:,-1]==s][:-1,:]
    outline[:,0] = convolve(outline[:,0],window,mode='wrap')
    outline[:,1] = convolve(outline[:,1],window,mode='wrap')
    offset = len(outline_0[outline_0[:,-1]<s])
    for i,p in enumerate(outline):
        f.write('Point({0:d}) = {{ {1:.1f},{2:.1f},0.0, lc }};\n'.format(i+offset,*p))

    for i in range(len(outline)):
        f.write('Line({0:d}) = {{ {1:d},{2:d} }};\n'.format(i+offset,i+offset,(i+1)%len(outline)+offset))

    f.write('Line Loop({0:d}) = {{'.format(s))
    for i in range(len(outline)-1):
        f.write('{0:d}, '.format(i+offset))
    f.write('{0:d} }};\n'.format(len(outline)-1 + offset))

    #f.write('Plane Surface({0:d}) = {{ {1:d} }};\n'.format(s,s))

f.write('Plane Surface(1) = {1};')
#f.write('Plane Surface(1) = {1};')#,2,3,4};')

f.close()

subprocess.call(["gmsh","-2","outline.geo"])
subprocess.call(["dolfin-convert","outline.msh","outline.xml"])

# Plot the resulting mesh
from dolfin import *
mesh = Mesh("outline.xml")
triplot(mesh.coordinates()[:,0],mesh.coordinates()[:,1],mesh.cells())





