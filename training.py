import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from utils import display
from utils import total
from utils import load_saved_data
from model import HQCFN

if __name__ == '__main__':

    torch.random.manual_seed(42)

    # Set paths for storing models and results
    model_path = './models/model_weights_mosi.pth'
    result_path = './results/result_mosi.csv'

    # Load datasets
    n_time_step = 20
    input_dims = [5, 20, 300]  # (audio_dim, visual_dim, text_dim)
    train_data, valid_data, test_data = load_saved_data(input_dims)

    # Set hyperparameters
    params = dict()
    params['n_ansatz_layers'] = [1, 2, 3]
    params['n_fusion_layers'] = [1, 2, 3, 4, 5]
    params['linear_lr'] = [0.001, 0.004, 0.008, 0.012]
    params['vqc_lr'] = [0.001, 0.004, 0.008, 0.012]
    params['batch_size'] = [16, 32, 64, 128]

    total_settings = total(params)
    seen_settings = set()
    print('There are {} different hyperparameter settings in total.'.format(total_settings))

    with open(result_path, 'w+', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(['n_ansatz_layers', 'n_fusion_layers', 'linear_lr', 'vqc_lr',
                         'batch_size', 'min_valid_mae', 'test_mae', 'test_corr',
                         'test_multi_acc', 'test_binary_acc', 'test_f1'])

    n_settings = 50
    for i in range(n_settings):
        n_ansatz_layers = random.choice(params['n_ansatz_layers'])
        n_fusion_layers = random.choice(params['n_fusion_layers'])
        linear_lr = random.choice(params['linear_lr'])
        vqc_lr = random.choice(params['vqc_lr'])
        batch_size = random.choice(params['batch_size'])

        # Reject the setting if it has been tried
        current_setting = (n_ansatz_layers, n_fusion_layers, linear_lr, vqc_lr, batch_size)
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        # Create data loaders
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

        # Initialize the model
        model = HQCFN(n_time_step, input_dims, n_ansatz_layers, n_fusion_layers)

        print('\nSetting {} model initialized.'.format(i))

        # Loss function and optimizer
        criterion = nn.L1Loss(reduction='sum')
        linear_a_params = list(map(id, model.linear_a.parameters()))
        linear_v_params = list(map(id, model.linear_v.parameters()))
        linear_t_params = list(map(id, model.linear_t.parameters()))
        quan_params = filter(lambda p: id(p) not in linear_a_params + linear_v_params + linear_t_params,
                             model.parameters())
        optimizer = torch.optim.Adam([
            {'params': model.linear_a.parameters(), 'lr': linear_lr},
            {'params': model.linear_v.parameters(), 'lr': linear_lr},
            {'params': model.linear_t.parameters(), 'lr': linear_lr},
            {'params': quan_params}], lr=vqc_lr)

        min_valid_loss = float('Inf')
        curr_patience = 20
        epochs = 200

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x_a, x_v, x_t, y in train_dataloader:
                pred = model([x_a, x_v, x_t])
                loss = criterion(pred, y)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_train_loss = train_loss / len(train_data)
            print('\nEpoch {}'.format(epoch))
            print('Training loss: {}'.format(avg_train_loss))

            model.eval()
            valid_loss = 0
            for x_a, x_v, x_t, y in valid_dataloader:
                pred_valid = model([x_a, x_v, x_t])
                valid_loss += criterion(pred_valid, y).item()
            avg_valid_loss = valid_loss / len(valid_data)
            print('Validation loss: {}'.format(avg_valid_loss))
            if avg_valid_loss < min_valid_loss:
                curr_patience = 20
                min_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), model_path)
                print('Found better model, saving to disk...')
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break

        best_model = HQCFN(n_time_step, input_dims, n_ansatz_layers, n_fusion_layers)
        best_model.load_state_dict(torch.load(model_path))
        best_model.eval()
        with torch.no_grad():
            x_a, x_v, x_t, y = next(iter(test_dataloader))
            pred_test = best_model([x_a, x_v, x_t])
        pred_test = pred_test.numpy()
        y = y.numpy()

        mae = np.mean(np.absolute(pred_test - y))
        corr = np.corrcoef(pred_test, y)[0][1]
        multi_acc = round(sum(np.round(pred_test) == np.round(y)) / float(len(y)), 5)
        true_label = (y >= 0)
        pred_label = (pred_test >= 0)
        bi_acc = accuracy_score(true_label, pred_label)
        f1 = f1_score(true_label, pred_label, average='weighted')
        display(mae, corr, multi_acc, bi_acc, f1)

        with open(result_path, 'a+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow([n_ansatz_layers, n_fusion_layers, linear_lr, vqc_lr,
                             batch_size, min_valid_loss, mae, corr, multi_acc,
                             bi_acc, f1])
