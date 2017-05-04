import kmodel.model as KM
import features as ft
import config as cnf

n_classes = 10
learn_rate = 0.001
training_iters = 500
batch_size = 50
display_step = 30

n_input = 64
n_steps = 80
n_hidden = 100


X_train, Y_train = ft.extract_mfcc_features(cnf.DATASET_BASE_PATH, cnf.TR_SUB_DIRS,
                                                  frames=n_steps, bands=n_input, add_channel=True)
Y_train = ft.one_hot_encode(Y_train)

X_test, Y_test = ft.extract_mfcc_features(cnf.DATASET_BASE_PATH, cnf.TST_SUB_DIRS,
                                                frames=n_steps, bands=n_input, add_channel=True)
Y_test = ft.one_hot_encode(Y_test)

params = {'loss': 'mse', 'optimizer': 'adam', 'learn_rate': learn_rate}
input_shape = (batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = KM.audio_crnn(params, nclass=n_classes, input_shape=input_shape)

for iter in xrange(training_iters):
        print("Iter: %s" % iter)
        offset = (iter * batch_size) % (Y_train.shape[0] - batch_size)
        batch_x = X_train[offset:(offset + batch_size), :, :]
        batch_y = Y_train[offset:(offset + batch_size), :]
        v = model.train_on_batch(batch_x, batch_y)

        if iter % display_step == 0:
            # Calculate accuracy
            v = model.evaluate(X_test, Y_test)
            print "Iter " + str(iter) + ", Minibatch Loss= " + \
                  "{:.6f}".format(v[0]) + ", Test Accuracy= " + \
                  "{:.5f}".format(v[1])

v = model.evaluate(X_test, Y_test)
print('Test loss: ', round(v[0]), 3)
print('Test accuracy: ', round(v[1]), 3)
model.save("crnn_%0.2f.pkl" % v[1])
# 46,69 best
