import os
from keras.utils.visualize_util import plot
import utils.urban_loader as uld
import tester as TST


def do_nn_train(model, train_gen, test_gen, cls_params, model_path=None):
    tr_hist, tst_hist, acc_hist = [], [], []
    for ep in xrange(cls_params['n_epoch']):
        print("Epoch %d" % ep)
        tr_loss, cnt = 0, 0
        for X_batch, Y_batch in train_gen.batch():
            v = model.train_on_batch(X_batch, Y_batch)
            tr_loss += v[0]
            cnt += 1
            print("Train loss: %.4f" % v[0])
        tr_hist.append(tr_loss / cnt)
        # ===
        # evaluate on test data, every epoch
        # ===
        tst_loss, tst_acc = TST.evaluate(model, test_gen)
        tst_hist.append(tst_loss)
        acc_hist.append(tst_acc)
        print('Train loss: %.4f' % tr_hist[-1])
        print('Test loss: %.4f' % tst_hist[-1])
        print('Test Acc: %.4f' % acc_hist[-1])
    # serialize model to HDF5
    model_file = os.path.join(model_path, 'crnn_%.4f_%.4f.h5' % (tst_hist[-1], acc_hist[-1]))
    model.save(model_file)
    plot(model, to_file=os.path.join(model_path, "model_graph.png"), show_shapes=True)
    # return tr_hist, tst_hist, peer_hist, eer_hist, best_model_name
