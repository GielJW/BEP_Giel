import os
import numpy as np
import matplotlib.pyplot as plt
import json
import operator
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def calc_expected_value(l, higher_better):
    values=[]
    l = np.array(l)
    d = {}
    for value in set(l):

        if higher_better==True:
            smaller_eq_prop = (sum(l <= value)) / len(l)
            smaller_prop = (sum(l < value)) / len(l)
        elif higher_better==False:
            smaller_eq_prop = (sum(l >= value)) / len(l)
            smaller_prop = (sum(l > value)) / len(l)


        tup = (smaller_eq_prop, smaller_prop)
        d[value]=tup

    stds=[]
    for j in range(1, len(l)):
        output = 0.0
        x2p = 0
        xp = 0
        #Iterate over every value
        for value in set(l):
            expected_value = value * ((d[value][0] ** j) - (d[value][1] ** j))
            output+=expected_value

            x2p += value**2 * ((d[value][0] ** j) - (d[value][1] ** j))
            xp += value * ((d[value][0] ** j) - (d[value][1] ** j))

        variance = x2p - xp**2

        values.append(output)
        stds.append(variance**0.5)

    return values, stds

#Find best hyperparameters
path='Evaluation_results_helpdesk_rnn/'
result_files = [filename for filename in os.listdir(path) if filename.startswith("suffix")]
best_loss_mae = 1000
best_loss_dls = 1000
best_experiment_mae = -1
best_experiment_dls = -1

losses_mae=[]
losses_dls=[]

expected_scores=[]

for file in result_files:
    with open('Evaluation_results_helpdesk_rnn/'+file, 'r') as f:
        parameters = json.load(f)
        loss_mae = parameters['rnn']['helpdesk.csv']['mae_denormalised']
        loss_dls = parameters['rnn']['helpdesk.csv']['dls']
        loss_mae = float(loss_mae)
        loss_dls = float(loss_dls)

    losses_mae.append(loss_mae)
    losses_dls.append(loss_dls)

    if loss_mae<best_loss_mae:
        best_loss_mae=loss_mae
        best_experiment_mae=file

    if loss_dls<best_loss_dls:
        best_loss_dls=loss_dls
        best_experiment_dls=file


losses_mae.sort()
losses_dls.sort()
rnn_losses_dls = losses_dls
rnn_losses_mae = losses_mae

print('worst MAE LSTM: ', losses_mae[-1])
print('median MAE LSTM: ', np.median(losses_mae))
print('mean MAE LSTM: ', np.mean(losses_mae))
print('best MAE LSTM: ', best_experiment_mae, best_loss_mae)
print("\n")
print('worst DLS LSTM: ', losses_dls[0])
print('median DLS LSTM: ', np.median(losses_dls))
print('mean DLS LSTM: ', np.mean(losses_dls))
print('best DLS LSTM: ', losses_dls[-1])
#
# #Boxplots
# plt.violinplot(losses_dls)
# plt.savefig('RNN_DLS_Box_helpdesk.png', format='png')
# plt.clf()
# plt.violinplot(losses_mae)
# plt.savefig('RNN_MAE_Box_helpdesk.png', format='png')
# plt.clf()
#

#Expected values
expected_DLS, stds0 = calc_expected_value(losses_dls, True)
plt.plot(expected_DLS)

ymin = list(map(operator.sub, expected_DLS, stds0))
ymin0 = [0 if x<0 else x for x in ymin]

ymax = list(map(operator.add, expected_DLS, stds0))
ymax0 = [expected_DLS[-1] if x>expected_DLS[-1] else x for x in ymax]

plt.fill_between(x=list(range(1,1000)), y1=ymin0,
                 y2=ymax0, alpha=0.2)
plt.savefig('RNN_DLS_helpdesk.png', format='png')
plt.clf()


expected_MAE, stds1 = calc_expected_value(losses_mae, False)
plt.plot(expected_MAE)

ymin = list(map(operator.sub, expected_MAE, stds1))
ymin1 = [expected_MAE[-1] if x<expected_MAE[-1] else x for x in ymin]

poly = np.polyfit(list(range(23,40)),ymin1[23:40],3)
ymin1[0:30] = np.poly1d(poly)(list(range(1,31)))


ymax1 = list(map(operator.add, expected_MAE, stds1))
ymax1 = [5000 if x>5000 else x for x in ymax1]

print(stds1)
print('ymim1: ', ymin1, '\n', 'ymax1: ', ymax1)

plt.fill_between(x=list(range(1,1000)), y1=ymin1,
                 y2=ymax1, alpha=0.2)
plt.ylim(3,6)
plt.savefig('RNN_MAE_helpdesk.png', format='png')
plt.clf()


print("\n ============================================ \n")

#Find best hyperparameters
path='Evaluation_results_helpdesk_ae/'
result_files = [filename for filename in os.listdir(path) if filename.startswith("suffix")]
best_loss_mae = 1000
best_loss_dls = 1000
best_experiment_mae = -1
best_experiment_dls = -1

losses_mae=[]
losses_dls=[]

expected_scores=[]
Lagrange=[]

for file in result_files:
    with open('Evaluation_results_helpdesk_ae/'+file, 'r') as f:
        parameters = json.load(f)
        loss_mae = parameters['ae']['helpdesk.csv']['mae_denormalised']
        loss_dls = parameters['ae']['helpdesk.csv']['dls']
        loss_mae = float(loss_mae)
        loss_dls = float(loss_dls)


    losses_mae.append(loss_mae)
    losses_dls.append(loss_dls)


    if loss_mae<best_loss_mae:
        best_loss_mae=loss_mae
        best_experiment_mae=file

    if loss_dls<best_loss_dls:
        best_loss_dls=loss_dls
        best_experiment_dls=file

losses_mae.sort()
losses_dls.sort()


print('worst MAE AE: ', losses_mae[-1])
print('median MAE AE: ', np.median(losses_mae))
print('mean MAE AE: ', np.mean(losses_mae))
print('best MAE AE: ', best_experiment_mae, best_loss_mae)
print("\n")
print('worst DLS AE: ', losses_dls[0])
print('median DLS AE: ', np.median(losses_dls))
print('mean DLS AE: ', np.mean(losses_dls))
print('best DLS AE: ', losses_dls[-1])

# #Boxplots
# plt.violinplot(losses_dls)
# plt.savefig('AE_DLS_Box_helpdesk.png', format='png')
# plt.clf()
# plt.violinplot(losses_mae)
# plt.savefig('AE_MAE_Box_helpdesk.png', format='png')
# plt.clf()
#
#Expected values
expected_DLS1, stds2 = calc_expected_value(losses_dls, True)
plt.plot(expected_DLS1)

ymin = list(map(operator.sub, expected_DLS1, stds2))
ymin2 = [0 if x<0 else x for x in ymin]
ymax = list(map(operator.add, expected_DLS1, stds2))
ymax2 = [expected_DLS1[-1] if x>expected_DLS1[-1] else x for x in ymax]

plt.fill_between(x=list(range(1,1000)), y1=ymin2,
                 y2=ymax2, alpha=0.2)
plt.savefig('AE_DLS_helpdesk.png', format='png')
plt.clf()



expected_MAE1, stds3 = calc_expected_value(losses_mae, False)
plt.plot(expected_MAE1)

ymin = list(map(operator.sub, expected_MAE1, stds3))
ymin3 = [expected_MAE1[-1] if x<expected_MAE1[-1] else x for x in ymin]

poly = np.polyfit(list(range(23,40)),ymin3[23:40],3)
ymin3[0:30] = np.poly1d(poly)(list(range(1,31)))

ymax3 = list(map(operator.add, expected_MAE1, stds3))
ymax3 = [5000 if x>5000 else x for x in ymax3]

plt.fill_between(x=list(range(1,1000)), y1=ymin3,
                 y2=ymax3, alpha=0.2)
plt.ylim(3,6)
plt.vlines([23, 40], ymin=0, ymax=1000)
plt.savefig('AE_MAE_helpdesk.png', format='png')
plt.clf()

#
#
#Combined
plt.plot(expected_DLS)
plt.plot(expected_DLS1)
red_patch = mpatches.Patch(color='blue', label='LSTM')
blue_patch = mpatches.Patch(color='orange', label='AE')
plt.legend(handles=[red_patch, blue_patch])

plt.fill_between(x=list(range(1,1000)), y1=ymin0,
                 y2=ymax0, alpha=0.2, color='blue')
plt.fill_between(x=list(range(1,1000)), y1=ymin2,
                 y2=ymax2, alpha=0.2, color='orange')
plt.xlabel('hyperparameter assignments')
plt.ylabel('expected best DLS')

plt.hlines(y=0.380, xmin=0, xmax=1200, colors='blue', linestyles='--', lw=1)
plt.hlines(y=0.388, xmin=0, xmax=1200, colors='orange', linestyles='--', lw=1)

plt.savefig('Combined_DLS_helpdesk_transferred.png', format='png')
plt.clf()


plt.plot(expected_MAE)
plt.plot(expected_MAE1)
red_patch = mpatches.Patch(color='blue', label='LSTM')
blue_patch = mpatches.Patch(color='orange', label='AE')
plt.legend(handles=[red_patch, blue_patch])

plt.fill_between(x=list(range(1,1000)), y1=ymin1,
                 y2=ymax1, alpha=0.2, color='blue')
plt.fill_between(x=list(range(1,1000)), y1=ymin3,
                 y2=ymax3, alpha=0.2, color='orange')

plt.hlines(y=4.36, xmin=0, xmax=1200, colors='blue', linestyles='--', lw=1)
plt.hlines(y=4.49, xmin=0, xmax=1200, colors='orange', linestyles='--', lw=1)

plt.xlabel('hyperparameter assignments')
plt.ylabel('expected best MAE')
plt.savefig('Combined_MAE_helpdesk_transferred.png', format='png')
plt.clf()



#Combined
plt.plot(expected_MAE)
plt.plot(expected_MAE1)
red_patch = mpatches.Patch(color='blue', label='LSTM')
blue_patch = mpatches.Patch(color='orange', label='AE')
plt.legend(handles=[red_patch, blue_patch])

plt.fill_between(x=list(range(1,1000)), y1=ymin1,
                 y2=ymax1, alpha=0.2, color='blue')
plt.fill_between(x=list(range(1,1000)), y1=expected_MAE1[-1],
                 y2=ymax3, alpha=0.2, color='orange')

plt.hlines(y=4.36, xmin=0, xmax=1200, colors='blue', linestyles='--', lw=1)
plt.hlines(y=4.49, xmin=0, xmax=1200, colors='orange', linestyles='--', lw=1)

plt.ylim(3,6)
plt.xlabel('hyperparameter assignments')
plt.ylabel('expected best MAE')
plt.savefig('YLIM_Combined_MAE_helpdesk_transferred.png', format='png')
plt.clf()
#
#
# plt.violinplot([rnn_losses_dls, losses_dls])
# plt.xticks([1, 2], ['LSTM', 'AE'])
# plt.savefig('Combined_Violin_DLS.png', format='png')
# plt.clf()
#
# plt.violinplot([rnn_losses_mae, losses_mae])
# plt.xticks([1, 2], ['LSTM', 'AE'])
# plt.savefig('Combined_Violin_MAE.png', format='png')
# plt.clf()
#


