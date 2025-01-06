import math
from typing import List
import numpy as np
import numpy.typing as npt

J_S = np.arange(4)                               # shape (4), [0, 1, 2, 3]
COMBS = np.array([math.comb(3, j) for j in J_S]) # shape (4), [1, 3, 3, 1]
class DependantsOfResolution_np:
    def __init__(self):
        self.nedges = 0
        self.v0: npt.NDArray[np.float64]
        self.u0: npt.NDArray[np.float64]
        self.v02: npt.NDArray[np.float64]
        self.u02: npt.NDArray[np.float64]
        self.uv_div: npt.NDArray[np.float64]
        self.berns: npt.NDArray[np.float64]
        self.berns2d: npt.NDArray[np.float64]
        self.dims1: npt.NDArray[np.float64]
        self.dims2: npt.NDArray[np.float64]
    
    def calculate_faces(self):
        faces: List[List[int]] = []
        for i in range(self.nedges-2):
            for j in range(self.nedges-2):
                face = [i * (self.nedges - 1) + j]
                face.append(i * (self.nedges - 1) + j + 1)
                face.append((i + 1) * (self.nedges - 1) + j + 1)
                face.append((i + 1) * (self.nedges - 1) + j)
                faces.append(face)
        self.faces = np.array(faces)

    def conditional_update(self, nedges: int):
        if self.nedges != nedges:
            self.nedges = nedges
            self.update()

    def update(self):
        v = np.linspace(0, 1, self.nedges + 1) # shape      (N+1)
        u = np.expand_dims(v, 1)               # shape (N+1, 1  )
        self.v0 = v[1:self.nedges]             # shape      (N-1)
        self.u0 = u[1:self.nedges]             # shape (N-1, 1  )
        self.v02 = np.expand_dims(np.stack((self.v0, 1-self.v0), 1), axis=(1,3)) # shape      (N-1, 1, 2, 1)
        self.u02 = np.expand_dims(self.v02, axis=4)                              # shape (N-1, 1  , 2, 1, 1)
        self.uv_div = self.u02 + self.v02                                        # shape (N-1, N-1, 2, 2, 1)

        self.berns = np.expand_dims(COMBS * self.u0**J_S * (1-self.u0)**(3-J_S), axis=1) # shape      (N-1, 1, 4)
        berns_np_2 = np.expand_dims(self.berns, axis=3)                                  # shape (N-1, 1  , 4, 1)
        self.berns2 = np.expand_dims(self.berns*berns_np_2, axis=4)                      # shape (N-1, N-1, 4, 4, 1)

        # numpy magic to get products of all combinations of Bernstein coefficients

        self.dims1 = np.empty((self.nedges-1, self.nedges-1, 1, 2, 3))
        self.dims2 = np.empty((self.nedges-1, self.nedges-1, 4, 1, 3))
        self.calculate_faces()