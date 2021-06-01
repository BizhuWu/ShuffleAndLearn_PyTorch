import matplotlib.pyplot as plt

loss_fromScratch = []
acc_fromScratch = []

loss_fromSSL = []
acc_fromSSL = []

with open('log_TrainOnUCFFromScratch.txt') as f:
    all_lines = f.readlines()

    for line in all_lines:
        if 'Loss' in line:
            loss_fromScratch.append(float(line[:-1].split(' ')[-1]))
        elif 'Acc' in line:
            acc_fromScratch.append(float(line[:-1].split(' ')[-1]))

with open('log_TrainOnUCFFromSSL.txt') as f:
    all_lines = f.readlines()

    for line in all_lines:
        if 'Loss' in line:
            loss_fromSSL.append(float(line[:-1].split(' ')[-1]))
        elif 'Acc' in line:
            acc_fromSSL.append(float(line[:-1].split(' ')[-1]))


plt.figure()
plt.plot(loss_fromScratch)
plt.plot(loss_fromSSL)
plt.title('Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(['From Scratch', 'From SSL'])
plt.show()

plt.figure()
plt.plot(acc_fromScratch)
plt.plot(acc_fromSSL)
plt.title('Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(['From Scratch', 'From SSL'])
plt.show()
