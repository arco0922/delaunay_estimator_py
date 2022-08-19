import numpy as np

# Calculate Jacobian by WLS(Weighted Least Squares) method 
def calc_jacobian(point, vector, adj_points, adj_vectors):
  input_dim = len(point)
  output_dim = len(vector)

  dr = [] if len(adj_points) == 0 else np.subtract(adj_points, point).tolist()
  dr_norms = [] if len(adj_points) == 0 else np.linalg.norm(dr, axis=1)
  weights = [1/x if x != 0 else 1 for x in dr_norms]

  dv = [] if len(adj_vectors) == 0 else np.subtract(adj_vectors, vector)

  _A = [[0]*input_dim for _ in [0]*input_dim]
  _B = [[0]*input_dim for _ in [0]*output_dim]

  for i_point in range(len(adj_points)):
    for i in range(input_dim):
      for j in range(input_dim):
        _A[i][j] += weights[i_point] * dr[i_point][j] * dr[i_point][i]
    
    for i in range(output_dim):
      for j in range(input_dim):
        _B[i][j] += weights[i_point] * dr[i_point][j] * dv[i_point][i]

  invA = np.linalg.pinv(_A)
  jacobian = np.matmul(_B, invA).tolist()

  return jacobian
