import os
import json
import datetime
import matplotlib.pyplot as plt

from cfg import cfg

metrics_filename = "{}_{}_{}_{}_metrics.json".format(cfg.noise_dataset, cfg.copy_model, cfg.initial_seed, cfg.ntest)
metrics_dir = os.path.join(cfg.noise_dataset, metrics_filename)


def save_metrics(budget=None, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
    if not os.path.exists(metrics_dir):
        metrics_dict = {
            "budget": [],
            "train_acc_list": [],
            "train_loss_list": [],
            "test_acc_list": [],
            "test_loss_list": []
        }
    else:
        with open(metrics_dir, 'r') as f:
            metrics_dict = json.load(f)

    budget_list = metrics_dict["budget"]
    train_acc_list = metrics_dict["train_acc_list"]
    train_loss_list = metrics_dict["train_loss_list"]
    test_acc_list = metrics_dict["test_acc_list"]
    test_loss_list = metrics_dict["test_loss_list"]

    if budget:
        budget_list.append(budget)

    if train_acc:
        train_acc = float(train_acc)
        train_acc_list.append(train_acc)

    if train_loss:
        train_loss = float(train_loss)
        train_loss_list.append(train_loss)

    if test_acc:
        test_acc = float(test_acc)
        test_acc_list.append(test_acc)

    if test_loss:
        test_loss = float(test_loss)
        test_loss_list.append(test_loss)

    with open(metrics_dir, 'w') as f:
        json.dump(metrics_dict, f, ensure_ascii=False)


def display_comparison_chart(train_metric_list, test_metric_list, budget_list,
                             title=None, axis_labels=None, line_labels=None, max_index_list=None):
    x = list(range(len(train_metric_list)))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    train_line = ax.plot(x, train_metric_list, color="deeppink",
                         linewidth=1, linestyle="-", label=line_labels[0])
    test_line = ax.plot(x, test_metric_list, color="darkblue",
                        linewidth=1, linestyle="-", label=line_labels[1])

    budget_ax = ax.twinx()
    budget_ax.set_ylabel("sample number", fontsize=14)
    budget_line = budget_ax.plot(x, budget_list, color="olive",
                                 linewidth=1, linestyle="-", label=line_labels[2])
    budget_ax.set_ylim(10000, 50000)

    lines = train_line + test_line + budget_line
    ax.legend(lines, line_labels, loc=0)

    for max_index in max_index_list:
        ax.plot(x[max_index], train_metric_list[max_index], color="deeppink", marker="P")
        ax.annotate("{:.4f}".format(train_metric_list[max_index]), xy=(x[max_index], train_metric_list[max_index]))
        ax.plot(x[max_index], test_metric_list[max_index], color="darkblue", marker="P")
        ax.annotate("{:.4f}".format(test_metric_list[max_index]), xy=(x[max_index], test_metric_list[max_index]))
        budget_ax.plot(x[max_index], budget_list[max_index], color="olive", marker="P")
        budget_ax.annotate(str(budget_list[max_index]), xy=(x[max_index], budget_list[max_index]))

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(axis_labels[0], fontsize=14)
    ax.set_ylabel(axis_labels[1], fontsize=14)
    ax.set_xticks(range(0, len(x) + 1, 10))

    filename = "_".join(title.replace(",", "").split(" "))
    current_time = "_".join(str(datetime.datetime.now()).split(".")[0].replace(":", "-").split(" "))
    filename = "_".join([filename, current_time]) + ".svg"
    plt.savefig(filename, format="svg")

    plt.show()



def get_max_index(metric_list):
    max_index_list = []
    max_metric = max(metric_list)
    for i in range(len(metric_list)):
        if metric_list[i] == max_metric:
            max_index_list.append(i)

    return max_index_list


def plot_metrics():
    assert os.path.exists(metrics_dir), "No metric saved!"
    with open(metrics_dir, 'r') as f:
        metrics_dict = json.load(f)

    budget_list = metrics_dict["budget"]
    train_acc_list = metrics_dict["train_acc_list"]
    train_loss_list = metrics_dict["train_loss_list"]
    test_acc_list = metrics_dict["test_acc_list"]
    test_loss_list = metrics_dict["test_loss_list"]


    assert len(budget_list) != 0, "No budget appended!"
    assert len(train_acc_list) != 0, "No acc appended!"
    assert len(train_loss_list) != 0, "No loss appended!"
    assert len(budget_list) == len(train_acc_list), "budget can't match to train acc"
    assert len(train_acc_list) == len(test_acc_list), "train acc can't match to test acc!"
    assert len(train_loss_list) == len(test_loss_list), "train loss can't match to test loss!"

    max_index_list = get_max_index(test_acc_list)

    title = "S1, S2 Comparison (NSFW, pretrained-imagenet-vgg16)"
    axis_labels = ["epoch", "similarity (accuracy)"]
    line_labels = ["S1", "S2", "budget"]
    display_comparison_chart(train_acc_list, test_acc_list, budget_list, title, axis_labels, line_labels, max_index_list)

    title = "Train Loss, Test Loss Comparison (NSFW, pretrained-imagenet-vgg16)"
    axis_labels = ["epoch", "loss"]
    line_labels = ["train loss", "test loss", "budget"]
    display_comparison_chart(train_loss_list, test_loss_list, budget_list, title, axis_labels, line_labels, max_index_list)


def clear_metrics():
    os.remove(metrics_dir)
