import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_utils import ScheduledWeightedSampler
from metrics import classify, accuracy, quadratic_weighted_kappa


def train(net, net_size, input_size, feature_dim, train_dataset, val_dataset,
          epochs, learning_rate, batch_size, save_path, pretrained_model=None):
    # create dataloader
    train_targets = [sampler[1] for sampler in train_dataset.imgs]
    weighted_sampler = ScheduledWeightedSampler(len(train_dataset), train_targets, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # define model
    model = net(net_size, input_size, feature_dim).cuda()
    print_msg('Trainable layers: ', ['{}\t{}'.format(k, v) for k, v in model.layer_configs()])

    # load pretrained weights
    if pretrained_model:
        pretrained_dict = model.load_weights(pretrained_model, ['fc', 'dense'])
        print_msg('Loaded weights from {}: '.format(pretrained_model), sorted(pretrained_dict.keys()))

    # define loss and optimizier
    MSELoss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    # learning rate decay
    milestones = [150, 220]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # train
    max_kappa = 0
    record_epochs, accs, losses = [], [], []
    model.train()
    for epoch in range(1, epochs + 1):
        # resampling weight update
        weighted_sampler.step()

        # learning rate update
        lr_scheduler.step()
        if epoch in milestones:
            print_msg('Learning rate decayed to {}'.format(lr_scheduler.get_lr()[0]))

        epoch_loss = 0
        correct = 0
        total = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, y = train_data
            X, y = X.cuda(), y.float().cuda()

            # forward
            y_pred = model(X)
            loss = MSELoss(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            total += y.size(0)
            correct += accuracy(y_pred, y) * y.size(0)
            avg_loss = epoch_loss / (step + 1)
            avg_acc = correct / total
            progress.set_description(
                'epoch: {}, loss: {:.6f}, acc: {:.4f}'
                .format(epoch, avg_loss, avg_acc)
            )

        # save model
        c_matrix = np.zeros((5, 5), dtype=int)
        acc = _eval(model, val_loader, c_matrix)
        kappa = quadratic_weighted_kappa(c_matrix)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
        if kappa > max_kappa:
            torch.save(model, save_path)
            max_kappa = kappa
            print_msg('Model save at {}'.format(save_path))

        # record
        record_epochs.append(epoch)
        accs.append(acc)
        losses.append(avg_loss)

    return record_epochs, accs, losses


def evaluate(model_path, test_dataset):
    c_matrix = np.zeros((5, 5), dtype=int)

    trained_model = torch.load(model_path).cuda()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_acc = _eval(trained_model, test_loader, c_matrix)
    print('========================================')
    print('Finished! test acc: {}'.format(test_acc))
    print('Confusion Matrix:')
    print(c_matrix)
    print('quadratic kappa: {}'.format(quadratic_weighted_kappa(c_matrix)))
    print('========================================')


def _eval(model, dataloader, c_matrix=None):
    model.eval()
    torch.set_grad_enabled(False)

    correct = 0
    total = 0
    for test_data in dataloader:
        X, y = test_data
        X, y = X.cuda(), y.float().cuda()

        y_pred = model(X)
        total += y.size(0)
        correct += accuracy(y_pred, y, c_matrix) * y.size(0)
    acc = round(correct / total, 4)

    model.train()
    torch.set_grad_enabled(True)
    return acc


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)
