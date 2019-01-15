import os
from datetime import datetime
from collections import OrderedDict

import click
import torch
import tqdm
import numpy as np
from torch import nn, optim
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from sentiment import utils
from sentiment import transforms as text_transforms
from sentiment import datasets
from sentiment.models import vdcnn

DATASETS = {
    'amazon_review_full': datasets.AmazonReviewFull,
    'amazon_review_polarity': datasets.AmazonReviewPolarity,
    'ag_news': datasets.AGNews,
    'dbpedia': datasets.DBPedia,
    'sogou_news': datasets.SogouNews,
    'yahoo_answers': datasets.YahooAnswers,
    'yelp_review_full': datasets.YelpReviewFull,
    'yelp_review_polarity': datasets.YelpReviewPolarity,
    'amazon': datasets.AmazonProductReviews,
}

MODELS = {
    'vdcnn9-conv': vdcnn.VDCNN9Conv,
    'vdcnn9-conv-shortcut': lambda num_classes=5: vdcnn.VDCNN9Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn9-maxpool': vdcnn.VDCNN9MaxPool,
    'vdcnn9-maxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN9MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn9-kmaxpool': vdcnn.VDCNN9KMaxPool,
    'vdcnn9-kmaxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN9KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501

    'vdcnn17-conv': vdcnn.VDCNN17Conv,
    'vdcnn17-conv-shortcut': lambda num_classes=5: vdcnn.VDCNN17Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn17-maxpool': vdcnn.VDCNN17MaxPool,
    'vdcnn17-maxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN17MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn17-kmaxpool': vdcnn.VDCNN17KMaxPool,
    'vdcnn17-kmaxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN17KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501

    'vdcnn29-conv': vdcnn.VDCNN29Conv,
    'vdcnn29-conv-shortcut': lambda num_classes=5: vdcnn.VDCNN29Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn29-maxpool': vdcnn.VDCNN29MaxPool,
    'vdcnn29-maxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN29MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn29-kmaxpool': vdcnn.VDCNN29KMaxPool,
    'vdcnn29-kmaxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN29KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501

    'vdcnn49-conv': vdcnn.VDCNN49Conv,
    'vdcnn49-conv-shortcut': lambda num_classes=5: vdcnn.VDCNN49Conv(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn49-maxpool': vdcnn.VDCNN49MaxPool,
    'vdcnn49-maxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN49MaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
    'vdcnn49-kmaxpool': vdcnn.VDCNN49KMaxPool,
    'vdcnn49-kmaxpool-shortcut': lambda num_classes=5: vdcnn.VDCNN49KMaxPool(num_classes=num_classes, shortcut=True),  # noqa: E501
}


def correct(outputs, targets, top=(1, )):
    _, predictions = outputs.topk(max(top), dim=1, largest=True, sorted=True)
    targets = targets.view(-1, 1).expand_as(predictions)

    corrects = predictions.eq(targets).cpu().int().cumsum(1).sum(0)
    tops = list(map(lambda k: corrects.data[k - 1], top))
    return tops


def run(epoch, model, loader, device, criterion=None, optimizer=None,
        top=(1, ), tracking=None, train=True):
    accuracies = [utils.AverageMeter() for _ in top]

    assert criterion is not None or not train, 'Need criterion to train model'
    assert optimizer is not None or not train, 'Need optimizer to train model'
    loader = tqdm.tqdm(loader)
    if train:
        model.train()
        losses = utils.AverageMeter()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        start = datetime.now()
        for batch_index, (inputs, targets) in enumerate(loader):
            batch_size = targets.size(0)
            assert batch_size < 2**32, 'Size is too large! correct will overflow'  # noqa: E501

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            if train:
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), batch_size)

            top_correct = correct(outputs, targets, top=top)
            for i, count in enumerate(top_correct):
                accuracies[i].update(count.item() * (100. / batch_size), batch_size)  # noqa: E501

            end = datetime.now()
            if tracking is not None:
                result = OrderedDict()
                result['timestamp'] = datetime.now()
                result['batch_duration'] = end - start
                result['epoch'] = epoch
                result['batch'] = batch_index
                result['batch_size'] = batch_size
                for i, k in enumerate(top):
                    result['top{}_correct'.format(k)] = top_correct[i].item()
                    result['top{}_accuracy'.format(k)] = accuracies[i].val
                if train:
                    result['loss'] = loss.item()
                utils.save_result(result, tracking)

            desc = 'Epoch {} {}'.format(epoch, '(Train):' if train else '(Val):  ')  # noqa: E501
            if train:
                desc += ' Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)  # noqa: E501
            for k, acc in zip(top, accuracies):
                desc += ' Prec@{} {acc.val:.3f} ({acc.avg:.3f})'.format(k, acc=acc)  # noqa: E501
            loader.set_description(desc)
            start = datetime.now()

    if train:
        message = 'Training accuracy of'
    else:
        message = 'Validation accuracy of'
    for i, k in enumerate(top):
        message += ' top-{}: {}'.format(k, accuracies[i].avg)
    print(message)
    return accuracies[0].avg


def create_graph(arch, timestamp, optimizer, restore,
                 learning_rate=None,
                 momentum=None,
                 weight_decay=None, num_classes=10):
    # create model
    model = MODELS[arch](num_classes=num_classes)

    # create optimizer
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))

    if restore is not None:
        if restore == 'latest':
            restore = utils.latest_file(arch)
        print(f'Restoring model from {restore}')
        assert os.path.exists(restore)
        restored_state = torch.load(restore)
        assert restored_state['arch'] == arch

        model.load_state_dict(restored_state['model'])

        if 'optimizer' in restored_state:
            optimizer.load_state_dict(restored_state['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

        best_accuracy = restored_state['accuracy']
        start_epoch = restored_state['epoch'] + 1
        run_dir = os.path.split(restore)[0]
    else:
        best_accuracy = 0.0
        start_epoch = 1
        run_dir = f"./run/{arch}/{timestamp}"

    print('Starting accuracy is {}'.format(best_accuracy))

    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))
    print(f"Run directory set to {run_dir}")

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))
    return run_dir, start_epoch, best_accuracy, model, optimizer


def create_test_dataset(dataset, dataset_dir, transform):
    test_dataset = DATASETS[dataset](root=dataset_dir, train=False,
                                     transform=transform)
    print(test_dataset)
    return test_dataset


def create_train_dataset(dataset, dataset_dir, transform):
    train_dataset = DATASETS[dataset](root=dataset_dir, train=True,
                                      transform=transform)
    print(train_dataset)
    return train_dataset


def _train(model, optimizer, criterion, device,
           train_loader, valid_loader, test_loader,
           start_epoch, epochs, learning_rates,
           track_test_acc, checkpoint, best_accuracy, run_dir, arch,
           use_cuda):
    train_results_file = os.path.join(run_dir, 'train_results.csv')
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')
    test_results_file = os.path.join(run_dir, 'test_results.csv')

    for nepochs, learning_rate in zip(epochs, learning_rates):
        end_epoch = start_epoch + nepochs
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        _lr_optimizer = utils.get_learning_rate(optimizer)
        if _lr_optimizer is not None:
            print('Learning rate set to {}'.format(_lr_optimizer))
            assert _lr_optimizer == learning_rate

        for epoch in range(start_epoch, end_epoch):
            run(epoch, model, train_loader, device, criterion, optimizer,
                tracking=train_results_file, train=True)

            valid_acc = run(epoch, model, valid_loader, device,
                            tracking=valid_results_file, train=False)

            if valid_loader != test_loader and track_test_acc:
                run(epoch, model, test_loader, device,
                    tracking=test_results_file, train=False)

            is_best = valid_acc > best_accuracy
            last_epoch = epoch == (end_epoch - 1)
            if is_best or checkpoint == 'all' or (checkpoint == 'last' and last_epoch):  # noqa: E501
                state = {
                    'epoch': epoch,
                    'arch': arch,
                    'model': (model.module if use_cuda else model).state_dict(),  # noqa: E501
                    'accuracy': valid_acc,
                    'optimizer': optimizer.state_dict()
                }
            if is_best:
                print('New best model!')
                filename = os.path.join(run_dir, 'checkpoint_best_model.t7')
                print(f'Saving checkpoint to {filename}')
                best_accuracy = valid_acc
                torch.save(state, filename)
            if checkpoint == 'all' or (checkpoint == 'last' and last_epoch):
                filename = os.path.join(run_dir, f'checkpoint_{epoch}.t7')
                print(f'Saving checkpoint to {filename}')
                torch.save(state, filename)

        start_epoch = end_epoch


@click.command()
@click.argument('dataset', type=click.Choice(DATASETS.keys()),
                default='amazon_review_full')
@click.option('--dataset-dir', default='./data')
@click.option('--checkpoint', '-c', type=click.Choice(['best', 'all', 'last']),
              default='last')
@click.option('--restore', '-r')
@click.option('--track-test-acc/--no-track-test-acc', default=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--epochs', '-e', multiple=True, default=[3, 3, 3, 3, 3],
              type=int)
@click.option('--batch-size', '-b', default=128)
@click.option('--learning-rates', '-l', multiple=True,
              default=[0.01, 0.005, 0.0025, 0.00125, 0.000625], type=float)
@click.option('--momentum', default=0.9)
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
              default='sgd')
@click.option('device_ids', '--device', '-d', multiple=True, type=int)
@click.option('--num-workers', type=int)
@click.option('--weight-decay', default=1e-4)
@click.option('--validation', '-v', default=0.0)
@click.option('--evaluate', is_flag=True)
@click.option('--shuffle/--no-shuffle', default=True)
@click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
              default='vdcnn9-maxpool')
def train(dataset, dataset_dir, checkpoint, restore, track_test_acc, cuda,
          epochs, batch_size, learning_rates, momentum, optimizer,
          device_ids, num_workers, weight_decay, validation, evaluate, shuffle,
          arch):
    timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
    local_timestamp = str(datetime.now())  # noqa: F841
    dataset_dir = os.path.join(dataset_dir, dataset + "_csv")
    config = {k: v for k, v in locals().items()}

    # Need dataset for number of classes and model defintion.
    #   Only loading test dataset so this is faster.
    print("Preparing {} test data".format(dataset.upper()))
    vocab = text_transforms.Vocab("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%ˆ&*~`+=<>()[]{} ",  # noqa: E501
                                  offset=2, unknown=1)
    transform_test = transforms.Compose([
        transforms.Lambda(lambda doc: doc.lower()),
        vocab,
        text_transforms.PadOrTruncate(1014),
        transforms.Lambda(lambda doc: doc.astype(np.int64))
    ])
    test_dataset = create_test_dataset(dataset, dataset_dir, transform_test)
    num_classes = test_dataset.classes

    # Need first learning rate for optimizer
    learning_rate = learning_rates[0]

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    run_dir, start_epoch, best_accuracy, model, optimizer = create_graph(
        arch, timestamp, optimizer, restore,
        learning_rate=learning_rate, momentum=momentum,
        weight_decay=weight_decay, num_classes=num_classes)

    utils.save_config(config, run_dir)

    # create loss
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if use_cuda:
        device_ids = device_ids or list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(
            model, device_ids=device_ids)
        num_workers = len(device_ids) if num_workers is None else num_workers
    else:
        num_workers = 0 if num_workers is None else num_workers
    print(f"using {num_workers} workers for data loading")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_cuda)

    if evaluate:
        print("Only running evaluation of model on test dataset")
        run(start_epoch - 1, model, test_loader, device,
            tracking=os.path.join(run_dir, 'test_results.csv'), train=False)
        return

    # load data
    print("Preparing {} training data:".format(dataset.upper()))
    transform_train = transform_test
    train_dataset = create_train_dataset(dataset, dataset_dir, transform_train)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    assert 1 > validation and validation >= 0, "Validation must be in [0, 1)"
    split = num_train - int(validation * num_train)

    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:split]
    valid_indices = indices[split:]

    print('Using {} examples for training'.format(len(train_indices)))
    print('Using {} examples for validation'.format(len(valid_indices)))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_cuda)
    if validation != 0:
        valid_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=valid_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=use_cuda)
    else:
        print('Using test dataset for validation')
        valid_loader = test_loader

    return _train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        start_epoch=start_epoch,
        epochs=epochs,
        learning_rates=learning_rates,
        track_test_acc=track_test_acc,
        checkpoint=checkpoint,
        best_accuracy=best_accuracy,
        run_dir=run_dir,
        arch=arch,
        use_cuda=use_cuda
    )


if __name__ == '__main__':
    train()
