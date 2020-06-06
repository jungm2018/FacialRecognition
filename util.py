# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# scikit-learn libraries
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#Extraneous

import time

#Imported Utility Functions
lfw_imageSize = (50,37)
def get_lfw_data(min_faces = 50) :
    """
    Fetch LFW (Labeled Faces in the Wild) dataset.
    
    Warning : This will take a long time the first time you run it.
    It will download data onto disk but then will use the local copy thereafter.
    
    Returns
    --------------------
        X -- numpy array of shape (n,d), features (each row is one image)
        y -- numpy array of shape (n,), targets
             elements are integers in [0, num_classes-1]
    """
    
    global X, n, d, y, h, w
    lfw_people = fetch_lfw_people(min_faces_per_person= min_faces, resize=0.4)
    n, h, w = lfw_people.images.shape
    X = lfw_people.data
    d = X.shape[1]
    y = lfw_people.target
    num_classes = lfw_people.target_names.shape[0]
 
    print("Total dataset size:")
    print("\tnum_samples: %d" % n)
    print("\tnum_features: %d" % d)
    print("\tnum_classes: %d" % num_classes)
    
    return X, y

def show_image(im, size=lfw_imageSize) :
    """
    Open a new window and display the image.
    
    Parameters
    --------------------
        im   -- numpy array of shape (d,), image
        size -- tuple (i,j), i and j are positive integers such that i * j = d
                default to the right value for LFW dataset
    """
    
    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap=cm.gray)
    plt.show()
##############################
#Custom Functions
##############################
def svd(A, singular_vector = False, verbose = False):
  '''
  Input: (m,n) matrix A
  Output: (m,m) matrix U, (m,n) matrix S, (n,n) matrix V
  '''
  m, n = np.shape(A)
  U, s, Vt = np.linalg.svd(A, full_matrices=True)
  S = np.zeros(np.shape(A))  
  S[:np.min((m,n)), :np.min((m,n))] = np.diag(s)
  if verbose:
    print("U dimensions: " + str(np.shape(U)))
    print("S dimensions: " + str(np.shape(S)))
    print("V dimensions: " + str(np.shape(Vt.T)))
  if singular_vector:
    return U, S, Vt.T, s
  return U, S, Vt.T

def truncated_svd(A, rank, verbose = True):
  '''
  Input: (m,n) matrix A, int rank
  Output: (m,k) matrix A_trun, ,(n,k) matrix Q
  '''
  U, S, V = svd(A, verbose = verbose)
  Q = V[:, 0:rank]
  A_trun = A.dot(Q)
  return A_trun, Q

def truncated_qr(A, trunc):
  '''
  Input: (m,n) matrix A, int trunc (rank)
  Output: (m,n) matrix Q, ,(n,n) matrix R
  '''
  m, n =  A.shape
  Q = np.copy(A); R = np.identity(n)
  for k in range(trunc):
    try:
      R[k, k] = np.linalg.norm(Q[:, k])
      Q[:, k] = Q[:, k] / R[k, k]
      R[k, k+1:n] = (Q[:, k].T).dot(Q[:, k+1:n])
      Q[:, k+1:n] = Q[:, k+1:n] - np.array([Q[:, k]]).T.dot(np.array([R[k, k+1:n]]))
    except Exception:
      print(A.shape, Q.shape, R.shape)
  return Q,R

def truncated_qr_reduction(A, trunc):
  '''
  Input: (m,n) matrix A, int trunc (rank)
  Output: (m,k) matrix A_trun, ,(n,k) matrix Q
  '''
  Q, R = truncated_qr(A.T, trunc)
  R_1 = R[0:trunc, 0:trunc]
  R_2 = R[0:trunc, trunc:]
  Q = Q[:, 0:trunc]
  A_trun = np.concatenate((R_1.T, R_2.T), axis = 0)
  return A_trun, Q

def k_means_reduction(A, trunc):
  '''
  Input: (m,n) matrix A, int trunc (rank)
  Output: (m,k) matrix A_trun, ,(n,k) matrix Q
  '''
  kmeans = KMeans(n_clusters = trunc, random_state=0, verbose = 0).fit(A)
  B_t = kmeans.cluster_centers_
  B = B_t.T
  C = np.zeros((X_train.shape[0], B_t.shape[0]))
  for m, label in enumerate(kmeans.labels_):
    C[m][label] = 1
  Q,R = np.linalg.qr(B)
  A_trun = (np.linalg.pinv(Q).dot(A.T)).T
  return A_trun, Q

def lda_reduction(A, y, trunc):
  '''
  Input: (m,n) matrix A, int trunc (rank)
  Output: (m,k) matrix A_trun, ,(n,k) matrix Q
  '''
  lda = LDA(n_components = trunc)
  A_trunc = lda.fit(A, y).transform(A)
  #Q = lda.scalings_[:, 0:trunc]
  Q = np.linalg.pinv(A).dot(A_trunc)
  return A_trunc, Q


def divide_into_classes(X, y):
  '''
  Input: (m,n) matrix X (data matrix),
          (m,) vector y (labels)
  Output: Map div, keys as labels, div[key] is a (data_per_label, n) matrix
  '''
  div = {}
  for n, label in enumerate(y):
    if label not in div.keys():
      div[label] = X[n, :]
    else:
      div[label] = np.vstack((div[label], X[n, :]))
  return div

def gaussian_likelihood(y, mu, cov):
  '''
  Input: (m,) matrix y, (m,) matrix mu, (m,m) matrix cov
  Output: numpy64 float, likelihood
  '''
  #omitting unecessary parts
  lam = np.log(np.linalg.eig(cov)[0])
  log_det = np.sum(lam)
  return (-1*((y-mu).T).dot(np.linalg.inv(cov)).dot(y-mu)-log_det)

def gaussian_likelihood_optimize(y, mu, cov_inv, log_cov_det):
  return (-1*((y-mu).T).dot(cov_inv).dot(y-mu)-log_cov_det)


def get_gaussian_param(div):
  '''
  Input: Map div, keys as labels, div[key] is a (data_per_label, n) matrix
  Output: (m,m) matrix cov_shared, map mean_k
  Note: Shared Covariance Matrix!!
  '''
  N_k = {}
  mean_k = {}
  cov_k = {}
  N = 0
  for key in div.keys():
    N_k[key] = div[key].shape[0]
    N = N + N_k[key]
    mean_k[key] = np.average(div[key], axis = 0)
    cov_k[key] = np.cov(div[key].T)
  cov_shared = sum([N_k[key]/N * cov_k[key] for key in div.keys()])
  return cov_shared, mean_k

def gaussian_classify(data, cov, mean_k, verbose = False):
  '''
  Input: (m,n) matrix data, (m,m) matrix cov, map mean_k
  Output: vector of classifications
  '''
  m, n = data.shape
  log_cov_det = np.sum(np.log(np.abs(np.linalg.eig(cov)[0])))
  cov_inv = np.linalg.inv(cov)
  classification = np.zeros(m)
  for i in range(m):
    if verbose and i%200 == 0:
      print(str(100*i/m) + '% Processed')
    likelihood = {}
    y = data[i][:]
    prediction, max_likelihood = -np.inf, -np.inf 
    for key in mean_k.keys():
      likelihood[key] = gaussian_likelihood_optimize(y, mean_k[key], cov_inv, log_cov_det)
      if likelihood[key] > max_likelihood:
        prediction, max_likelihood = key, likelihood[key]
    classification[i] = prediction
  return classification

def center(A):
  print(A.shape)
  A_ = np.copy(A)
  for i in range(A.shape[1]):
    A_[:, i] = A_[:, i] - np.average(A_[:, i])
  return A_

def get_accuracy(y, y_class):
  temp = y - y_class
  num_points = temp.shape[0]
  num_right = 0
  for i in range(num_points):
    num_right = num_right + 1 if temp[i] == 0 else num_right
  accuracy = num_right/num_points
  return accuracy

def get_test_reduced_datamatrix(X_test, Q):
  return X_test.dot(Q)

def train_test(X_train, X_test, y_train, y_test, sigma, type = 'svd'):
  start = time.time()
  if type == 'svd':
    X_trun_train, Q = truncated_svd(X_train, sigma, verbose = False)
  elif type == 'qr':
    X_trun_train, Q = truncated_qr_reduction(X_train, sigma)
  elif type =='kmeans':
    X_trun_train, Q = k_means_reduction(X_train, sigma)
  elif type == 'LDA':
    X_trun_train, Q = lda_reduction(X_train, y_train, sigma)
  else:
    raise ValueError(type + ' is not a valid dimensionality reduction technique')
  end = time.time(); dt = end - start
  print(type + ' ----- Factorization Time: ' + str(dt))
  training_data = divide_into_classes(X_trun_train, y_train)
  cov, mean = get_gaussian_param(training_data)
  
  y_class = gaussian_classify(X_trun_train, cov, mean, verbose = False)
  train_accuracy = get_accuracy(y_train, y_class)
  X_trun_test = get_test_reduced_datamatrix(X_test, Q)
  y_class = gaussian_classify(X_trun_test, cov, mean, verbose = False)
  test_accuracy = get_accuracy(y_test, y_class)
  return train_accuracy, test_accuracy
