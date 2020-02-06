import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import BDRRN, BDRRN_cat
from dataset import mDataset, ETEDataset
from utils import AverageMeter
from torch.optim.lr_scheduler import StepLR

cudnn.benchmark = True
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='BDRRN')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--parts_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=False, default='')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--format', type=str, required=False, default='.png')
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda_device', type=int, default=2)
    parser.add_argument('--use_pretrained', action='store_true')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)


    if opt.arch == 'BDRRN':
        model = BDRRN(opt.num_channels)
    elif opt.arch == 'BDRRN_cat':
        model = BDRRN_cat(opt.num_channels)
    else:
        raise KeyError()

    if opt.use_pretrained:
        if opt.weights_path!='':
            state_dict = model.state_dict()
            weights =  torch.load(opt.weights_path)['model_state_dict']
            model.load_state_dict(weights)
            #for n, p in torch.load(weights, map_location=lambda storage, loc: storage).items():
            #    if n in state_dict.keys():
            #        state_dict[n].copy_(p)
            #    else:
            #        raise KeyError(n)
        else:
            raise KeyError()

    cudnn.benchmark = True
    device_name = 'cuda:' + str(opt.cuda_device)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()


    if opt.arch == 'BDRRN':
         #optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
         optimizer = optim.Adam([
             {'params': model.conv1.parameters()},
             {'params': model.conv2.parameters()},
             {'params': model.input.parameters()},
             {'params': model.relu.parameters()},
             {'params': model.bn.parameters()},
             {'params': model.output.parameters(), 'lr': opt.lr * 0.1},
         ], lr=opt.lr)
    elif opt.arch == 'BDRRN_cat':
        optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.convcat.parameters()},
            {'params': model.input.parameters()},
            {'params': model.relu.parameters()},
            {'params': model.bn.parameters()},
            {'params': model.output.parameters(), 'lr': opt.lr * 0.1},
        ], lr=opt.lr)
    else:
         optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.use_pretrained:
        weights = torch.load(opt.weights_path)['optimizer_state_dict']
        optimizer.load_state_dict(weights)


    dataset = mDataset(opt.images_dir, opt.parts_dir, opt.labels_dir, opt.num_channels )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + opt.start + 1, opt.num_epochs+ opt.start))
            for data in dataloader:
                ## modify for double branch
                inputs, parts, labels = data
                inputs = inputs.to(device)
                parts = parts.to(device)
                labels = labels.to(device)

                preds = model(inputs, parts)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        torch.save({
            'epoch': epoch + opt.start,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_losses.avg},
            os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch + opt.start)))
