import os
import yaml
import random
import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
from transformers import BertForSequenceClassification

from tqdm import tqdm
from sklearn.model_selection import KFold

from utils.schedule import WarmupConstantSchedule
from utils.plot_results import plot_curve, plot_cm
from dataloader_text import dataloader_bert

torch.manual_seed(1234)
random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


def train(net, train_iterator, optimizer, criterion):
    net.train()
    epoch_loss = 0.0
    epoch_corrects = 0

    for batch in train_iterator:
        inputs = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = net(
            inputs,
            token_type_ids=None,
            attention_mask=input_mask,
            labels=labels
        )
        loss = criterion(outputs.logits, labels)

        _, predict = torch.max(outputs.logits, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(predict == labels.data)

    train_loss = epoch_loss / len(train_iterator.dataset)
    train_acc = epoch_corrects.double().item() / len(train_iterator.dataset)

    return net, train_loss, train_acc


def valid(net, valid_iterator, criterion):
    net.eval()
    epoch_loss = 0.0
    epoch_corrects = 0

    target_lists = torch.zeros(0, dtype=torch.long, device='cpu')
    predict_lists = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for batch in valid_iterator:
            inputs = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = net(
                inputs,
                token_type_ids=None,
                attention_mask=input_mask,
                labels=labels
            )
            loss = criterion(outputs.logits, labels)

            _, predict = torch.max(outputs.logits, 1)

            target_lists = torch.cat([target_lists, labels.cpu()])
            predict_lists = torch.cat([predict_lists, predict.cpu()])

            epoch_loss += loss.item() * inputs.size(0)
            epoch_corrects += torch.sum(predict == labels.data)

        valid_loss = epoch_loss / len(valid_iterator.dataset)
        valid_acc = epoch_corrects.double().item() / len(valid_iterator.dataset)

    return net, valid_loss, valid_acc, predict_lists.numpy(), target_lists.numpy()


if __name__ == "__main__":
    # ================== [1] Set up ==================
    # load param
    with open('hyper_param.yaml', 'r') as file:
        config = yaml.safe_load(file.read())

    in_path = config['dataset_setting']['in_path']
    max_len = config['dataset_setting']['max_len']

    epochs = config['training_setting']['epoch']
    batch_size = config['training_setting']['batch_size']
    learning_rate = config['training_setting']['learning_rate']
    warmup_rate = config['training_setting']['warmup_rate']
    early_stopping = config['training_setting']['early_stopping']

    cross_validation = 0.0
    cv_lists = []

    # make dir for results
    time_now = datetime.datetime.now()
    os.makedirs(
        "./results/{}/accuracy_curve".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/confusion_matrix".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/learning_curve".format(str(time_now.date())), exist_ok=True)
    os.makedirs(
        "./results/{}/model_param".format(str(time_now.date())), exist_ok=True)

    fold = KFold(n_splits=5, shuffle=True, random_state=0)

    dataloader = dataloader_bert(max_len=max_len, in_path=in_path)
    input_ids_train, attention_masks_train, labels_train = dataloader.get_train_data()
    train_dataset = data.TensorDataset(
        input_ids_train, attention_masks_train, labels_train)

    # ================== [2] Training and Validation ==================
    print("Start training!")
    for fold_idx, (train_idx, valid_idx) in tqdm(enumerate(fold.split(train_dataset))):
        net = BertForSequenceClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            num_labels=4,
            output_attentions=False,
            output_hidden_states=False
        )

        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate
        )
        scheduler = WarmupConstantSchedule(
            optimizer,
            warmup_epochs=epochs * warmup_rate
        )

        net.to(device)
        if fold_idx == 0:
            print(net)

        # Freezing pretrained parameter
        for i, param in enumerate(net.parameters()):
            if i < 199:
                param.requires_grad = False

        train_acc_curve = []
        valid_acc_curve = []
        train_loss_curve = []
        valid_loss_curve = []

        print('fold:{}\n'.format(fold_idx))

        train_iterator = data.DataLoader(
            data.Subset(train_dataset, train_idx),
            shuffle=True,
            batch_size=batch_size
        )

        valid_iterator = data.DataLoader(
            data.Subset(train_dataset, valid_idx),
            shuffle=False,
            batch_size=batch_size
        )

        patience = 0
        for epoch in tqdm(range(epochs)):
            print("train phase")
            net, train_loss, train_acc = train(
                net, train_iterator, optimizer, criterion)
            print("test phase")
            net, valid_loss, valid_acc, predict_list, target_list = valid(
                net, valid_iterator, criterion)

            scheduler.step()

            print(
                'epoch {}/{} | train_loss {:.8f} train_acc {:.8f} | valid_loss {:.8f} valid_acc {:.8f}'.format(
                    epoch+1, epochs, train_loss, train_acc, valid_loss, valid_acc
                ))

            # ===== early-stopling =====
            if train_loss < valid_loss:
                patience += 1
                if patience > early_stopping:
                    break
            else:
                patience = 0
            # ==========================

            train_loss_curve.append(train_loss)
            train_acc_curve.append(train_acc)
            valid_loss_curve.append(valid_loss)
            valid_acc_curve.append(valid_acc)

        torch.save(
            net.state_dict(),
            './results/{}/model_param/SER_JTES_TEXT_fold{}_Param_{}.pth'.format(
                str(time_now.date()), fold_idx, time_now)
        )
        cv_lists.append(valid_acc)
        cross_validation += valid_acc

        plot_curve(
            train_acc_curve,
            valid_acc_curve,
            x_label=config['plot_acc_curve_setting']['acc_curve_x_label'],
            y_label=config['plot_acc_curve_setting']['acc_curve_y_label'],
            title=config['plot_acc_curve_setting']['acc_curve_title'],
            fold_idx=fold_idx,
            dir_path_name=str(time_now.date())
        )
        plot_curve(
            train_loss_curve,
            valid_loss_curve,
            x_label=config['plot_loss_curve_setting']['loss_curve_x_label'],
            y_label=config['plot_loss_curve_setting']['loss_curve_y_label'],
            title=config['plot_loss_curve_setting']['loss_curve_title'],
            fold_idx=fold_idx,
            dir_path_name=str(time_now.date())
        )
        plot_cm(
            target_list,
            predict_list,
            x_label=config['plot_cm_setting']['cm_x_label'],
            y_label=config['plot_cm_setting']['cm_y_label'],
            dir_path_name=str(time_now.date())
        )

    # ================== [3] Plot results of CV ==================
    print("cross validation:{}".format(cv_lists))
    print("cross validation [ave]:{}".format(cross_validation / len(cv_lists)))
    print("Finished!")
