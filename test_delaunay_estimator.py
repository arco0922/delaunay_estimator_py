# %%
from delaunay_estimator.delaunay_estimator import DelaunayEstimator
import numpy as np
# %%
delaunay = DelaunayEstimator(2,1)

# %%
delaunay.add_point([0,0], [1])

#%%
delaunay.add_point([0,1], [2])

# %%
delaunay.add_point([1,0], [3])

# %%
delaunay.add_point([1,1], [4])

# %%
delaunay.add_point([2,3], [5])

# %%
delaunay.add_point([1,3], [6])

# %%
print(delaunay.simplex_point_ids)
print(delaunay.simplex_transforms)
print(delaunay.hull_facet_point_ids)
print(delaunay.hull_facet_jacobians)
print(delaunay.simplex_vectors)

# %%

print(delaunay.suggest_vectors([0.25,0.25]))
print(delaunay.suggest_vectors([100,100]))

# %%
