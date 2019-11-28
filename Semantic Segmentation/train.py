# training the model
from tqdm.autonotebook import tqdm
import numpy as np
import torch
from utils import *
import matplotlib.pyplot as plt

def train(args, net, trainloader, valloader, criterion,optimizer ):
    train_loss = []
    val_loss = []
    train_iou = []
    val_iou = []
    best_loss = np.inf
    for epoch in tqdm(range(1, args.num_epochs+1)):
        tl = []
        vl = []
        preds = []
        targets = []
        net.train()
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            predicted = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels.type(torch.cuda.LongTensor))
            tl.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds.extend(list(predicted.cpu().numpy()))
            targets.extend(list(labels.cpu().numpy()))

        t = np.mean(tl)
        train_loss.append(t)

        # calculate iou and dice for train
        preds = np.array(preds)
        targets = np.array(targets)
        cm = get_confusion_matrix(preds.flatten(), targets.flatten())
        tr_iou, _ = evaluate(cm)
        train_iou.append(tr_iou)

        # perform validation
        net.eval()
        preds = []
        targets = []
        for (images, labels) in valloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            loss = criterion(outputs, labels.type(torch.cuda.LongTensor))
            predicted = torch.argmax(outputs, 1)
            preds.extend(list(predicted.cpu().numpy()))
            targets.extend(list(labels.cpu().numpy()))
            vl.append(loss.item())
        v = np.mean(vl)
        val_loss.append(v)

        # calculate iou and dice for val
        preds = np.array(preds)
        targets = np.array(targets)
        cm = get_confusion_matrix(preds.flatten(), targets.flatten())
        vl_iou, _ = evaluate(cm)
        val_iou.append(vl_iou)

        # saving the best model
        if v < best_loss:
            best_loss = v
            model_dict = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(model_dict, args.best_model+'.pth')

        print(f"Epoch[{epoch}/{args.num_epochs}], Train loss: {t: .4f}, loss: {v: .4f}, Train IOU: {tr_iou: .4f}, Val IOU: {vl_iou: .4f}")

    # loss vs epoch plot
    plt.plot(val_loss, color='green', label='Validation loss')
    plt.plot(train_loss, color='blue', label='Train loss')
    plt.legend(loc='upper right', shadow=True, fontsize='large')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.savefig(args.best_model+'.png')