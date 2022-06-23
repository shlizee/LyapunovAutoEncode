import torch
from AEPredNet import AEPredNet
from AE_utils import mini_batch_ae, train_val_split
import matplotlib.pyplot as plt
import os
import argparse

def ae_train(model, train_data, train_targets, val_data, val_targets, batch_size = 64, val_batch_size=64, alpha= 1, epochs = 100,
                verbose = True, print_interval = 10, suffix = '', device = torch.device('cpu')):
    train_loss  = torch.zeros((epochs,), device = device)
    val_loss 	= torch.zeros((epochs,), device = device)
    val1 		= torch.zeros((epochs,), device = device)
    val2 		= torch.zeros((epochs,), device = device)
    model.to(device)
    for epoch in range(epochs):
        x_train = mini_batch_ae(train_data, batch_size)
        tar_train = mini_batch_ae(train_targets, batch_size)
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
            train_batches+= 1
            temp, _ = model.train_step_ae(input = xt.to(device), targets = tt.to(device), alpha = alpha)
            tl += float(temp) * train_samples/batch_size
        for xv, tv in zip(x_val, tar_val):
            val_samples = xv.shape[0]
            val_batches+= 1
            losses = model.val_step_ae(input = xv.to(device), targets =tv.to(device), alpha = alpha)
            vl+= float(losses[0])* val_samples/val_batch_size
            vl1+= float(losses[1])* val_samples/val_batch_size
            vl2 += float(losses[2])* val_samples/val_batch_size
        # print(torch.Tensor([vl/val_samples], device = device))
        val_loss[epoch] = vl/val_batches
        # print(torch.mean(torch.Tensor(vl1)).shape)
        val1[epoch] = vl1/val_batches
        val2[epoch] = vl2/val_batches
        train_loss[epoch] = tl/train_batches

        if val2[epoch] < model.best_val:
            model.best_val = val2[epoch]
            model.best_state = model.state_dict()
        model.global_step +=1
        if verbose and epoch%print_interval == 0:
            # print(model.val_loss)
            print(f'Validation Loss at epoch {model.global_step -1}: {val_loss[epoch]:.3f}')
            print(f'Best Prediction Loss: {model.best_val:.3f}')
    model.train_loss = torch.cat((model.train_loss, train_loss.cpu()))
    model.val_loss = torch.cat((model.val_loss, val_loss.cpu()))
    model.vl1 = torch.cat((model.vl1, val1.cpu()))
    model.vl2 = torch.cat((model.vl2, val2.cpu()))
    model.alphas = torch.cat((model.alphas, torch.Tensor([alpha, epochs]).unsqueeze(dim = 0)), dim = 0)
    if not os.path.exists(f'Models/Latent_{model.latent_size}'):
        os.mkdir(f'Models/Latent_{model.latent_size}')
    torch.save(model, f'Models/Latent_{model.latent_size}/ae_prednet_{model.global_step}{suffix}.ckpt')


def main(args):
    parser = argparse.ArgumentParser(description="Train Lyapunov Autoencoder")
    parser.add_argument("-model", "--model_type", type=str, default= 'lstm', required=False)
    parser.add_argument("-task", "--task_type", type=str, default='SMNIST', required=False)
    parser.add_argument("-latent", "--latent_size", type=int, default= 32, required=False)
    parser.add_argument("-alphas", "--alphas", type=str, default= '[5, 5, 10, 20]', required=False)
    parser.add_argument("-ae_lr", "--lr", type=float, default= 1e-5, required=False)
    parser.add_argument("-epochs", "--epochs", type=int, default= 1000, required=False)

    args = parser.parse_args(args)
    model_type = args.model_type
    task_type = args.task_type
    latent_size = args.latent_size
    temp = ''

    for ch in args.alphas:
        temp += ch
    epochs = args.epochs
    lr = args.lr
    temp = temp[1:-1].split(',')
    alphas = []
    for alpha in temp:
        alphas.append(int(alpha))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dir = f'Processed/trials/{task_type}/{model_type}'
    x_data = torch.load(f'{dir}/{model_type}_allLEs.p')
    targets = torch.load(f'{dir}/{model_type}_allValLoss.p')
    if os.path.exists(f'Processed/{model_type}_data_split_vfrac0.2.p'):
        split = torch.load(f'Processed/{model_type}_data_split_vfrac0.2.p')
    else:
        split = train_val_split(x_data, targets, 0.2)
    x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'], split['val_data'], split['val_targets']
    if not os.path.isdir('Models/'):
        os.mkdir('Models')
    model = AEPredNet(latent_size = latent_size, lr = lr, act = 'tanh', device = device)
    for alpha in alphas:
        ae_train(model, x_train, y_train, x_val, y_val, alpha = alpha, epochs = epochs, print_interval = 250, batch_size = 128, device = device)
    plt.plot(range(model.global_step), model.val_loss, label = 'total')
    plt.plot(range(model.global_step), model.vl1, label ='rec loss (L1)')
    plt.plot(range(model.global_step), model.vl2, label = 'pred loss (L2)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    if not os.path.isdir('Figures/'):
        os.mkdir('Figures')
    plt.savefig(f"Figures/{model_type}_Prednet_AE_valCurve.png",bbox_inches="tight",dpi=200)



if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
