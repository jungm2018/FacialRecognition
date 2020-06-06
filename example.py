from util import *

X, y = get_lfw_data(min_faces = 30)
print(X.shape)
X = center(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
singular_values = np.logspace(1, 2.8, num = 50, dtype='int32')
svd_train_accuracy = []; svd_test_accuracy = []
qr_train_accuracy = []; qr_test_accuracy = []
k_means_train_accuracy = []; k_means_test_accuracy = []
for sigma in singular_values:
  print('Truncated: ' + str(sigma))
  svd_train_accuracy_, svd_test_accuracy_ = train_test(X_train, X_test, y_train, y_test,
                                                       sigma, type = 'svd')
  qr_train_accuracy_, qr_test_accuracy_ = train_test(X_train, X_test, y_train, y_test,
                                                       sigma, type = 'qr')
  k_means_train_accuracy_, k_means_test_accuracy_ =  train_test(X_train, X_test, y_train, y_test,
                                                       sigma, type = 'kmeans')

  svd_train_accuracy.append(svd_train_accuracy_); svd_test_accuracy.append(svd_test_accuracy_)
  qr_train_accuracy.append(qr_train_accuracy_); qr_test_accuracy.append(qr_test_accuracy_)
  k_means_train_accuracy.append(k_means_train_accuracy_); k_means_test_accuracy.append(k_means_test_accuracy_)

plt.plot(singular_values, svd_train_accuracy, color = 'r', marker = 'o')
plt.plot(singular_values, svd_test_accuracy, color = 'b', marker = 'o')
plt.plot(singular_values, qr_train_accuracy, color = 'r', marker = 'x')
plt.plot(singular_values, qr_test_accuracy, color = 'b', marker = 'x')
plt.plot(singular_values, k_means_train_accuracy, color = 'r', marker = '+')
plt.plot(singular_values, k_means_test_accuracy, color = 'b', marker = '+')
