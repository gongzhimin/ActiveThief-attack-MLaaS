import numpy as np
import time
from cfg import cfg, config
from dsl.dsl_marker_v2 import DSLMarker, collect_aux_data
from sss.random_sss import RandomSelectionStrategy
from sss.adversarial_sss import AdversarialSelectionStrategy
from sss.uncertainty_sss import UncertaintySelectionStrategy
from sss.kcenter_sss import KCenterGreedyApproach

import tensorflow as tf
import logging
from utils.dataset_utils import get_true_model_predictions, load_imgs
from utils.visualization import save_metrics


def compute_accuracy(model, sess, dsl):
    num_batches = dsl.get_num_batches()

    t_loss= []
    t_acc = []

    dsl.reset_batch_counter()

    for step in range(num_batches):
        X, Y = dsl.load_next_batch()

        X = load_imgs(X)

        Yhat, loss, _, global_step = sess.run(
            [
                model.pred,
                model.loss,
                model.train_op,
                model.global_step
            ],
            feed_dict={
                model.imgs: X,
                model.true_out: Y
            }
        )
        Y = np.argmax(Y, axis=1)
        agreement_count = np.sum(Yhat == Y)
        t_acc += [agreement_count / float(len(Y))]
        t_loss += [loss]

    return np.mean(t_acc), np.mean(t_loss)


def get_predictions(sess, model, x, one_hot=False, labels=False):
    Y = []
    Y_prob = []
    Y_idx = []

    for start in range(0, len(x), cfg.batch_size):
        X = x[start:start + cfg.batch_size]

        X = load_imgs(X)

        Y_b, Y_prob_b, Y_idx_b = sess.run(
            [
                model.pred_one_hot,
                model.probs,
                model.pred
            ],
            feed_dict={model.imgs: X}
        )

        Y.append(Y_b)
        Y_prob.append(Y_prob_b)
        Y_idx.append(Y_idx_b)

    Y = np.concatenate(Y)
    Y_prob = np.concatenate(Y_prob)
    Y_idx = np.concatenate(Y_idx)

    if one_hot:
        if labels:
            return Y, Y_idx
        else:
            return Y
    else:
        if labels:
            return Y_prob, Y_idx
        else:
            return Y_prob


# For KCenter
def get_initial_centers(sess, noise_train_dsl_marked, copy_model):
    Y_vec_true = []

    noise_train_dsl_marked.reset_batch_counter()
    for b in range(noise_train_dsl_marked.get_num_batches()):
        trX, _ = noise_train_dsl_marked.load_next_batch()
        trY = get_predictions(sess, copy_model, trX, labels=False)
        Y_vec_true.append(trY)

    Y_vec_true = np.concatenate(Y_vec_true)

    return Y_vec_true


# new train iter
def train_copynet_iter(copy_model, train_dsl, valid_dsl, test_dsl, sess):
    """ Trains the copy_model iteratively"""
    num_classes = train_dsl.get_num_classes()

    budget = cfg.initial_seed + cfg.ntest * num_classes
    print "budget: ", budget
    logging.info("budget: {}".format(budget))

    train_dsl_marker = DSLMarker(train_dsl)
    train_dsl_marked, train_dsl_unmarked = train_dsl_marker.get_split_dsls()

    train_time = time.time()

    # Mark initial samples
    train_label_counts = dict(list(enumerate([0] * num_classes)))
    for i in range(cfg.initial_seed):
        train_dsl_marker.mark(i)
        label = train_dsl.label_dict[train_dsl.data[i]]
        train_label_counts[label] += 1
    print "initial train label class dist: ", train_label_counts
    logging.info("initial train label class dist: {}".format(train_label_counts))

    test_label_counts = dict(list(enumerate([0] * num_classes)))
    for label in test_dsl.labels:
        test_label_counts[label] += 1
    print "test label class dist: ", test_label_counts
    logging.info("test label class dist: {}".format(test_label_counts))

    pred_match = []

    for it in range(cfg.num_iter + 1):
        print "Processing iteration ", it + 1
        logging.info("Processing iteration {}".format(it + 1))

        label_counts = dict(list(enumerate([0] * num_classes)))

        copy_model_acc, _ = compute_accuracy(copy_model, sess, test_dsl)
        print 'copy model acc', copy_model_acc
        logging.info('copy model acc: {}'.format(copy_model_acc))

        exit = False
        best_acc = None
        no_improvement = 0

        for epoch in range(cfg.copy_num_epochs):
            if it == 2 and epoch == 5:
                copy_model.save_weights(sess)
            t_loss = []
            t_acc = []
            epoch_time = time.time()

            print "\nProcessing epoch {} of iteration {}".format(epoch + 1, it + 1)
            logging.info("\nProcessing epoch {} of iteration {}".format(epoch + 1, it + 1))

            train_dsl_marked.reset_batch_counter()
            train_dsl_marker.shuffle_data()

            global_step = 0
            for i in range(train_dsl_marked.get_num_batches()):
                trX, trY = train_dsl_marked.load_next_batch(return_idx=False, return_aux=False)

                trX = load_imgs(trX)

                trYhat, loss, _, global_step = sess.run(
                    [
                        copy_model.pred,
                        copy_model.loss,
                        copy_model.train_op,
                        copy_model.global_step
                    ],
                    feed_dict={
                        copy_model.imgs: trX,
                        copy_model.true_out: trY
                    }
                )

                if epoch == 0:
                    for class_ in list(np.argmax(trY, -1)):
                        label_counts[class_] += 1

                trY = np.argmax(trY, axis=1)
                agreement_count = np.sum(trYhat==trY)

                t_acc += [agreement_count / float(len(trY))]
                t_loss += [loss]

            train_loss = np.mean(t_loss)
            train_acc = np.mean(t_acc)
            print('Epoch: {} Step: {}'.format(epoch + 1, global_step))
            logging.info('Epoch: {} \tStep: {}'.format(epoch + 1, global_step))

            print "Train Acc: {} \tTrain Loss: {}".format(train_acc, train_loss)
            logging.info("Train Acc: {} \tTrain Loss: {}".format(train_acc, train_loss))

            test_acc, test_loss = compute_accuracy(copy_model, sess, test_dsl)
            print 'Test accuracy: {} \tTest loss: {}'.format(test_acc, test_loss)
            logging.info('Test accuracy: {} \tTest loss: {}'.format(test_acc, test_loss))

            if best_acc is None or test_acc > best_acc:
                best_acc = test_acc
                print "[BEST]",
                logging.info("[BEST]"),

                no_improvement = 0
            else:
                no_improvement += 1

                if (no_improvement % cfg.early_stop_tolerance) == 0:
                    if train_loss > 1.7:
                        no_improvement = 0
                    else:
                        exit = True

            save_metrics(budget=budget,
                         train_acc=train_acc, train_loss=train_loss,
                         test_acc=test_acc, test_loss=test_loss)


            print "End of epoch {} (took {} minutes).".format(epoch + 1, round((time.time() - epoch_time) / 60, 2))
            logging.info(
                "End of epoch {} (took {} minutes).".format(epoch + 1, round((time.time() - epoch_time) / 60, 2)))

            if exit:
                print "Number of epochs processed: {} in iteration {}".format(epoch + 1, it + 1)
                logging.info("Number of epochs processed: {} in iteration {}".format(epoch + 1, it + 1))
                break

        if it + 1 == cfg.num_iter + 1:
            break

        X = []
        Y = []
        Y_idx = []
        idx = []

        train_dsl_unmarked.reset_batch_counter()

        print train_dsl_unmarked.get_num_batches()

        for b in range(train_dsl_unmarked.get_num_batches()):
            trX, _, tr_idx = train_dsl_unmarked.load_next_batch(return_idx=True)

            for jj in tr_idx:
                assert jj not in train_dsl_marker.marked_set, "MASSIVE FAILURE!!"

            trY, trY_idx = get_predictions(sess, copy_model, trX, labels=True)

            X.append(trX)
            Y.append(trY)
            Y_idx.append(trY_idx)
            idx.append(tr_idx)

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Y_idx = np.concatenate(Y_idx)
        idx = np.concatenate(idx)

        for jj in idx:
            assert jj not in train_dsl_marker.marked_set, "MASSIVE FAILURE 2!!"

        sss_time = time.time()
        # Core Set Construction
        if cfg.sampling_method == 'random':
            sss = RandomSelectionStrategy(cfg.k, Y)
        elif cfg.sampling_method == 'adversarial':
            sss = AdversarialSelectionStrategy(cfg.k, Y, X, sess, copy_model, K=len(Y))
        elif cfg.sampling_method == 'uncertainty':
            sss = UncertaintySelectionStrategy(cfg.k, Y)
        elif cfg.sampling_method == 'kcenter':
            sss = KCenterGreedyApproach(cfg.k, Y, get_initial_centers(sess, train_dsl_marked, copy_model))
        elif cfg.sampling_method == 'adversarial-kcenter':
            sss = AdversarialSelectionStrategy(budget, Y, X, sess, copy_model, K=len(Y))
            s2 = np.array(sss.get_subset())
            sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, train_dsl_marked, copy_model))
        else:
            raise Exception("sampling method {} not implemented".format(cfg.sampling_method))

        s = sss.get_subset()

        if cfg.sampling_method in ['adversarial-kcenter']:
            s = s2[s]

        print "{} selection time:{} min".format(cfg.sampling_method, round((time.time() - sss_time) / 60, 2))
        logging.info("{} selection time:{} min".format(cfg.sampling_method, round((time.time() - sss_time) / 60, 2)))

        if cfg.sampling_method != 'kmeans' and cfg.sampling_method != 'kcenter':
            assert len(s) == cfg.k

        print "len(s):", len(s)
        print "len(unique(s)):", len(np.unique(s))
        logging.info("len(s): {}".format(len(s)))
        logging.info("len(unique(s)): {}".format(len(np.unique(s))))

        s = np.unique(s)
        budget += len(s)

        pred_true_count = np.zeros((num_classes, num_classes), dtype=np.int32)

        trX = [X[e] for e in s]

        true_trY, true_trY_idx = get_true_model_predictions(trX, train_dsl)
        foobXs = dict()
        foobYs = dict()
        foobYps = dict()

        train_dsl_marked.reset_batch_counter()
        for b in range(train_dsl_marked.get_num_batches()):
            foobX, foobY, foobI = train_dsl_marked.load_next_batch(return_idx=True)
            _, foobYp = get_true_model_predictions(foobX, train_dsl)

            for idx1, foobIdx in enumerate(foobI):
                foobXs[foobIdx] = foobX[idx1]
                foobYps[foobIdx] = foobYp[idx1]

        print "Mark count before:", len(train_dsl_marker.marked)
        logging.info("Mark count before: {}".format(len(train_dsl_marker.marked)))

        for jj in idx:
            assert jj not in train_dsl_marker.marked_set, "MASSIVE FAILURE 3!!"

        for i, k in enumerate(s):
            train_dsl_marker.mark(idx[k])
            pred_true_count[true_trY_idx[i]][Y_idx[k]] += 1

        train_dsl_marked.reset_batch_counter()
        not_found_count = 0
        hit_count = 0
        for b in range(train_dsl_marked.get_num_batches()):
            foobX, foobY, foobI = train_dsl_marked.load_next_batch(return_idx=True)
            _, foobYp = get_true_model_predictions(foobX, train_dsl)

            for idx1, foobIdx in enumerate(foobI):
                if foobIdx in foobXs:
                    hit_count += 1
                    assert foobXs[foobIdx] == foobX[idx1]
                    assert foobYps[foobIdx] == foobYp[idx1]

                    del foobXs[foobIdx]
                else:
                    not_found_count += 1

        print "Mark count after:", len(train_dsl_marker.marked)
        print "Not found count:", not_found_count
        print "Found count:", hit_count
        print "Found unique:", len(foobYs) - len(foobXs)
        print "Did not find unique:", len(foobXs)
        print "Prediction agreement between source and copy model on selected subset is {}/{}".format(
            np.trace(pred_true_count), len(pred_true_count))

        logging.info("Mark count after: {}".format(train_dsl_marker.marked))
        logging.info("Not found count: {}".format(not_found_count))
        logging.info("Found count: {}".format(hit_count))
        logging.info("Found unique: {}".format(len(foobYs) - len(foobXs)))
        logging.info("Did not find unique: {}".format(len(foobXs)))
        logging.info("Prediction agreement between source and copy model on selected subset is {}/{}".format(
            np.trace(pred_true_count), len(pred_true_count)))

        pred_match.append(pred_true_count)

        print "End of iteration ", it + 1
        logging.info("End of iteration {}".format(it + 1))

    if pred_match:
        pred_match = np.stack(pred_match, axis=0)
        logging.info("pred_match: {}".format(pred_match))

    print "Copynet training completed in {} time".format(round((time.time() - train_time) / 3600, 2))
    print "---Copynet trainning completed---"


