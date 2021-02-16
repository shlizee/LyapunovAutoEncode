import torch
from AEPredNet import AEPredNet
from AE_utils import mini_batch_ae, train_val_split
import matplotlib.pyplot as plt
import os
import pickle

def ae_train(model, train_data, train_targets, val_data,
             val_targets, batch_size=64, val_batch_size=64, alpha=1,
             epochs=100, verbose=True, print_interval=10,
             suffix='', device=torch.device('cpu'), inputs_epoch=4):
    train_loss = torch.zeros((epochs,), device=device)
    val_loss = torch.zeros((epochs,), device=device)
    val1 = torch.zeros((epochs,), device=device)
    val2 = torch.zeros((epochs,), device=device)
    for epoch in range(epochs):
        x_train = mini_batch_ae(train_data, batch_size)
        tar_train = mini_batch_ae(train_targets, batch_size)
        # print("tar_train: ", tar_train)
        x_val = mini_batch_ae(val_data, val_batch_size)
        tar_val = mini_batch_ae(val_targets, val_batch_size)
        tl = 0
        vl = 0
        vl1 = 0
        vl2 = 0
        train_batches = 0
        val_batches = 0
        for xt, tt in zip(x_train, tar_train):
            train_samples = xt.shape[0]
            train_batches += 1
            loss, outputs = model.train_step_ae(input=xt.to(device), targets=tt.to(device),alpha=alpha)
            tl += float(loss * train_samples / batch_size)
        for xv, tv in zip(x_val, tar_val):
            val_samples = xv.shape[0]
            val_batches += 1
            losses = model.val_step_ae(input=xv.to(device), targets=tv.to(device), alpha=alpha)
            vl += float(losses[0]) * val_samples / val_batch_size
            vl1 += float(losses[1]) * val_samples / val_batch_size
            vl2 += float(losses[2]) * val_samples / val_batch_size
        # print(torch.Tensor([vl/val_samples], device = device))
        val_loss[epoch] = vl / val_batches
        # print(torch.mean(torch.Tensor(vl1)).shape)
        val1[epoch] = vl1 / val_batches
        val2[epoch] = vl2 / val_batches
        train_loss[epoch] = tl / train_batches

        if val2[epoch] < model.best_val:
            model.best_val = val2[epoch]
            model.best_state = model.state_dict()
        model.global_step += 1
        if verbose and epoch % print_interval == 0:
            # print(model.val_loss)
            print(f'Validation Loss at epoch {model.global_step - 1}: {val_loss[epoch]:.3f}')
            print(f'Best Prediction Loss: {model.best_val:.3f}')
    model.train_loss = torch.cat((model.train_loss, train_loss))
    model.val_loss = torch.cat((model.val_loss, val_loss))
    model.vl1 = torch.cat((model.vl1, val1))
    model.vl2 = torch.cat((model.vl2, val2))
    model.alphas = torch.cat((model.alphas, torch.Tensor([alpha, epochs]).unsqueeze(dim=0)), dim=0)
    torch.save(model, f'Results/N_512_g_1.5/epoch_{inputs_epoch}/{model.prediction_loss_type}/ae_prednet_{model.global_step}{suffix}.ckpt')


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    N = 512
    g = 1.5
    inputs_epoch = 7
    target_epoch = 14
    val_split = 0.1
    data_path = "training_data/g_1.5/4sine_epoch_{}_N_{}".format(inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))
    x_data, targets = data['inputs'], data['targets']

    split = train_val_split(data=x_data, targets=targets, val_split=val_split)

    torch.save(split, f'Results/N_{N}_g_{g}/epoch_{inputs_epoch}/data_split_vfrac{val_split}.p')

    x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'], split['val_data'], split[
        'val_targets']
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    prediction_loss_type = "MSE"
    model = AEPredNet(latent_size=128, lr=1e-4, act='tanh', device=device, prediction_loss_type=prediction_loss_type)
    alphas = [5,5,10, 20]
    # alphas = [20, 20, 30, 40]


    for alpha in alphas:
        ae_train(model, x_train, y_train, x_val, y_val, alpha=alpha, epochs=1000, print_interval=250, batch_size=128,
                 inputs_epoch=inputs_epoch)
    plt.plot(range(model.global_step), model.val_loss, label='total')
    plt.plot(range(model.global_step), model.vl1, label='rec loss (L1)')
    plt.plot(range(model.global_step), model.vl2, label='pred loss (L2)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig("../lyapunov-hyperopt-master/Figures/Prednet_AE_valCurve_epoch_{}.png".format(inputs_epoch), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
