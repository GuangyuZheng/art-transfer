import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f, True)


def get_average_precision(file_path):
    if os.path.isfile(file_path) is False:
        return 'Not tested!'
    sum = 0.0
    num = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split()) == 0:
                continue
            precision = float(line.split()[-1])
            sum += precision
            num += 1
    if num == 0:
        return 'Not tested!'
    else:
        return sum/num


def sentiment_analysis_show_acc(model, test_input, testY, batch_size=16):
    writer = open("error.txt", "w", encoding="utf-8")

    testZZ = model.predict(test_input, batch_size=batch_size)

    testZ = []

    for line in testZZ:
        maxx = 0
        ans = 0
        ll = line

        if ll > 0.5:
            ans = 1
        else:
            ans = 0
        testZ.append(ans)

    TP = 0
    PP = 0
    for i in range(len(testY)):
        PP += 1
        wtf = testY[i]

        if testZ[i] == wtf:
            TP += 1
        else:
            writer.write(str(testZZ[i]) + " " + str(testY[i]))
            writer.write("\n")
    writer.close()
    print('TP/PP', TP, PP)
    prec = TP / PP
    print('P : ', prec)
    return prec


def pos_tagging_show_acc(model, test_input, testY, tag="", batch_size=128):
    # print(time.clock())
    testZZ = model.predict(test_input, batch_size=batch_size)
    testZ = []
    for line in testZZ:
        tem = []
        for k, tag in enumerate(line):
            maxTag = 0
            maxScore = 0
            for kk, score in enumerate(tag):
                if maxScore < score:
                    maxScore = score
                    maxTag = kk
            tem.append(maxTag)
        testZ.append(tem)
    # print(len(testZZ))
    # print(time.clock())
    TP = 0
    PP = 0
    RR = 0
    lenTest = len(test_input[0])
    # print(lenTest)
    for i in range(lenTest):
        outputY = []
        outputX = []
        for k, tag in enumerate(testZ[i]):
            if max(testY[i][k]) != 0:
                if testY[i][k][tag] == 1:
                    TP += 1
                # outputY.append(id2tg[tag - 1])
                # outputX.append(id2ch[testX[i][k] - 1])
                PP += 1
            else:
                break
    print(tag)
    prec = TP / PP
    print('TP/PP={}/{}=P:{}'.format(TP, PP, prec))
    return prec


def plot_matrix(cm,
                x_labels,
                y_labels,
                normalize=False,
                title='Attention matrix',
                cmap=plt.cm.Blues, file_name='attention.pdf'):
    """
    This function prints and plots the attention matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized attention matrix")
    else:
        print('Attention matrix, without normalization')

    # print(cm)
    pdf = PdfPages(filename=file_name)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)

    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)

    map = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cb = plt.colorbar(mappable=map)

    figure.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()