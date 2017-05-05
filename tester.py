import numpy as np


def evaluate(model, test_gen, metrics=['accuracy']):
    # calc evaluation scores
    loss, acc, cnt = 0, 0, 0
    for tst_file, X_tst_b, Y_tst_b in test_gen.batch():
        l = model.test_on_batch(X_tst_b, Y_tst_b)
        loss += l[0]
        acc += l[1]
        cnt += 1
    avg_loss = loss / cnt
    avg_acc = acc / cnt
    return avg_loss, avg_acc

        # ps = model.predict_on_batch(X_tst_b)
        # average scores across sequential chunks from utterance
        # avg_ps = np.mean(ps, axis=0)
        # collect evaluation scores
        # for tag, sc, sc_fx in zip(avg_ps):
        #     eval_scores.append((tst_file, tag, sc))
        #     fix_scores.append((tst_file, tag, sc_fx))
        # loss val
        # l = model.test_on_batch(X_tst_b, Y_tst_b)
        # loss += l[0]
        # cnt += 1
    # loss /= cnt
