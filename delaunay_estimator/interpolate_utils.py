import numpy as np

def linear_interpolate_vector(ratio, vectors):
  num = len(ratio)
  if num == 0:
    return []
  if num != len(vectors):
    print('length of ratio and vectors must be equal')
    raise ValueError
      
  dim = len(vectors[0])
  res_vector = [0]*dim

  for i in range(dim):
    for n in range(num): 
      res_vector[i] += ratio[n] * vectors[n][i]

  return res_vector


def linear_interpolate_matrix(ratio, matrixes):
  num = len(ratio)
  if num == 0:
    return []
  if num != len(matrixes):
    print('length of ratio and matrixes must be equal')
    raise ValueError

  dim1 = len(matrixes[0])
  dim2 = len(matrixes[0][0])

  res_matrix = [[0]*dim2 for _ in [0]*dim1]

  for i in range(dim1):
    for j in range(dim2):
      for n in range(num):
        res_matrix[i][j] += ratio[n] * matrixes[n][i][j]

  return res_matrix


def first_order_approximation(vector, jacobian, dr):
  res = np.add(vector, np.matmul(jacobian, dr)).tolist()
  return res
