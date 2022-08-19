import numpy as np
import scipy.spatial

from delaunay_estimator.grad_utils import calc_jacobian
from delaunay_estimator.interpolate_utils import linear_interpolate_vector, linear_interpolate_matrix, first_order_approximation

class DelaunayEstimator:
  def __init__(self, input_dim, output_dim): 
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.points = []
    self.vectors = []
    self.simplex_point_ids = []
    self.simplex_vectors = []
    self.hull_facet_point_ids = []
    self.hull_facet_vectors = []
    self.simplex_transforms = []
    self.hull_facet_jacobians = []


  def add_point(self, coord, vector): 
    self.points.append(coord)
    self.vectors.append(vector)
    self._update_delaunay()
    self._update_simplex_transforms()
    self._update_hull_facet_jacobians()


  def estimate(self, coord): 
    i_simplex, barycentric_coord_in_simplex = self._find_simplex(coord)
    if (i_simplex >= 0):
      res = linear_interpolate_vector(
        barycentric_coord_in_simplex,
        self.simplex_vectors[i_simplex],
      )
      return res

    i_facet, barycentric_coord_of_nearest_point, ordinary_coord_of_nearest_point = self._find_nearest_facet(coord)
    vector_at_facet = linear_interpolate_vector(barycentric_coord_of_nearest_point, self.hull_facet_vectors[i_facet])
    jacobian_at_facet = linear_interpolate_matrix(barycentric_coord_of_nearest_point, self.hull_facet_jacobians[i_facet])
    dr = np.subtract(coord, ordinary_coord_of_nearest_point).tolist()
    
    res = first_order_approximation(vector_at_facet, jacobian_at_facet, dr)
    return res


  def suggest_vectors(self, coord):
    i_simplex, barycentric_coord_in_simplex = self._find_simplex(coord)
    if (i_simplex >= 0):
      base_simplex_vector = self.simplex_vectors[i_simplex][self.input_dim]
      suggestion_vectors_info = []
      for i in range(self.input_dim):
        diff_simplex_vector = np.subtract(self.simplex_vectors[i_simplex][i], base_simplex_vector).tolist()
        min_limit = -barycentric_coord_in_simplex[i]
        max_limit = 1 - barycentric_coord_in_simplex[i]
        suggestion_vectors_info.append({
          'vector': diff_simplex_vector,
          'min_coefficient': min_limit,
          'max_coefficient': max_limit,
        })
      return suggestion_vectors_info

    i_facet, barycentric_coord_of_nearest_point, _ = self._find_nearest_facet(coord)
    jacobian_at_facet = linear_interpolate_matrix(barycentric_coord_of_nearest_point, self.hull_facet_jacobians[i_facet])
    suggestion_vectors_info = []
    for i in range(self.input_dim):
      dr = [0]*self.input_dim
      dr[i] = 1.0
      diff_vector = np.matmul(jacobian_at_facet, dr).tolist()
      suggestion_vectors_info.append({
        'vector': diff_vector,
        'min_coefficient': -2,
        'max_coefficient': 2,
      })
    return suggestion_vectors_info


  def _update_delaunay(self):
    if (len(self.points) <= self.input_dim):
      self.simplex_point_ids = []
      self.hull_facet_point_ids = [[i for i in range(len(self.points))]]
    else:
      _delaunay = scipy.spatial.Delaunay(self.points)
      self.simplex_point_ids = _delaunay.simplices.tolist()
      self.hull_facet_point_ids = _delaunay.convex_hull.tolist()
    
    if len(self.simplex_point_ids) > 0:
      self.simplex_vectors = np.array(self.vectors)[np.array(self.simplex_point_ids)].tolist()
    if len(self.hull_facet_point_ids) > 0:
      self.hull_facet_vectors = np.array(self.vectors)[np.array(self.hull_facet_point_ids)].tolist()


  def _update_simplex_transforms(self):
    simplex_transforms = []
    for i_simplex in range(len(self.simplex_point_ids)):
      i_simplex_transform = [[0]*self.input_dim for _ in [0]*(self.input_dim+1)]
      for j in range(self.input_dim):
        i_simplex_transform[self.input_dim][j] = self.points[self.simplex_point_ids[i_simplex][self.input_dim]][j]
      _T = [[0]* self.input_dim for _ in [0]*self.input_dim]
  
      for i in range(self.input_dim):
        for j in range(self.input_dim):
          _T[i][j] = self.points[self.simplex_point_ids[i_simplex][j]][i] - self.points[self.simplex_point_ids[i_simplex][self.input_dim]][i]

      inv_T = np.linalg.pinv(_T).tolist()
      for i in range(self.input_dim):
        for j in range(self.input_dim):
          i_simplex_transform[i][j] = inv_T[i][j]
      
      simplex_transforms.append(i_simplex_transform)

    self.simplex_transforms = simplex_transforms


  def _update_hull_facet_jacobians(self):
    hull_facet_jacobians = []
    for i_facet in range(len(self.hull_facet_point_ids)):
      i_facet_jacobian = []
      hull_points = self.hull_facet_point_ids[i_facet]

      for i in range(len(hull_points)):
        hull_point_id = hull_points[i]
        adj_points_ids = self._find_adj_point_ids(hull_point_id)
        adj_points = []
        adj_vectors = []
        for p_id in adj_points_ids:
          adj_points.append(self.points[p_id])
          adj_vectors.append(self.vectors[p_id])

        jacobi = calc_jacobian(
          self.points[hull_point_id],
          self.vectors[hull_point_id],
          adj_points,
          adj_vectors,
        )
        i_facet_jacobian.append(jacobi)

      hull_facet_jacobians.append(i_facet_jacobian)

    self.hull_facet_jacobians = hull_facet_jacobians


  def _find_adj_point_ids(self, point_id):
    adj_points = set()

    if len(self.simplex_point_ids) == 0:
      for p_id in range(len(self.points)):
        if p_id != point_id:
          adj_points.add(p_id)

    else:
      for simplex in self.simplex_point_ids:
        _simplex = set(simplex)
        if point_id in _simplex:
          for p_id in _simplex:
            if p_id != point_id:
              adj_points.add(p_id)

    return adj_points


  def _find_simplex(self, coord):
    eps = 0.00001
    for i_simplex in range(len(self.simplex_point_ids)):
      barycentric_coord_in_simplex = self._calc_barycentric_coord_in_simplex(coord, i_simplex)
      is_inside = True
      
      for i in range(len(barycentric_coord_in_simplex)):
        if not -eps <= barycentric_coord_in_simplex[i] <= 1+eps:
          is_inside = False

      if is_inside:
        return i_simplex, barycentric_coord_in_simplex

    return -1, []


  def _find_nearest_facet(self, coord):
    best_dist = float('inf')
    best_i_facet = -1
    best_barycentric_coord_of_nearest_point = []
    best_ordinary_coord_of_nearest_point = []

    for i_facet in range(len(self.hull_facet_point_ids)):
      barycentric_coord_of_nearest_point = self._calc_barycentric_coord_of_nearest_point_in_facet(coord, i_facet)
      ordinary_coord_of_nearest_point = self._calc_ordinary_coord_from_barycentric_coord_in_facet(barycentric_coord_of_nearest_point, i_facet)
      dist = np.linalg.norm(np.subtract(coord, ordinary_coord_of_nearest_point))
      
      if dist < best_dist:
        best_dist = dist
        best_i_facet = i_facet
        best_barycentric_coord_of_nearest_point = barycentric_coord_of_nearest_point
        best_ordinary_coord_of_nearest_point = ordinary_coord_of_nearest_point

    return best_i_facet, best_barycentric_coord_of_nearest_point, best_ordinary_coord_of_nearest_point


  def _calc_barycentric_coord_of_nearest_point_in_facet(self, coord, i_facet):
    num_points_in_i_facet = len(self.hull_facet_point_ids[i_facet])

    if (num_points_in_i_facet == 1):
      return [1.0]

    U = [np.subtract(
      self.points[self.hull_facet_point_ids[i_facet][idx]],
      self.points[self.hull_facet_point_ids[i_facet][num_points_in_i_facet - 1]]).tolist() for idx in range(num_points_in_i_facet-1)]

    r = np.subtract(
      coord,
      self.points[self.hull_facet_point_ids[i_facet][num_points_in_i_facet - 1]]).tolist()

    A = [[0]*(num_points_in_i_facet-1) for _ in [0]*(num_points_in_i_facet-1)]
    for i in range(num_points_in_i_facet - 1):
      for j in range(num_points_in_i_facet - 1):
        A[i][j] = np.dot(U[i], U[j])

    b = [np.dot(U[i], r) for i in range(num_points_in_i_facet-1)]

    C = np.matmul(np.linalg.inv(A),b).tolist()

    c = [0]*num_points_in_i_facet
    c[num_points_in_i_facet - 1] = 1.0
    for i in range(i < num_points_in_i_facet - 1):
      c[i] = min(max(C[i], 0), 1)
      c[num_points_in_i_facet - 1] -= c[i]

    return c


  def _calc_ordinary_coord_from_barycentric_coord_in_facet(self, barycentric_coord, i_facet):
    facet_points = np.array(self.points)[self.hull_facet_point_ids[i_facet]].tolist()
    ordinary_coord = linear_interpolate_vector(barycentric_coord, facet_points)
    return ordinary_coord


  def _calc_barycentric_coord_in_simplex(self, coord, i_simplex):
    dim = len(coord)
    C = [0]*(dim+1)
    C[dim] = 1.0
    transform = self.simplex_transforms[i_simplex]
    for i in range(dim):
      for j in range(dim):
        C[i] += transform[i][j] * (coord[j] - transform[dim][j])
      C[dim] -= C[i]
    
    return C
