from keras.datasets import mnist
from keras.models import Sequential, load_model
mnist_model = load_model('mnist_r.h5') # load the saved model
mnist_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
(X_train, y_train), (X_test, y_test) = mnist.load_data()
loss_and_metrics = mnist_model.evaluate(X_test, y_test, verbose=2)

