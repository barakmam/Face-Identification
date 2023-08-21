# train nets like in the reference paper https://link.springer.com/content/pdf/10.1007/s10462-021-09974-2.pdf?pdf=button%20sticky
# to reproduce results
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from insightFace.recognition.arcface_torch.backbones import get_model
from ArcFace_paulpias.model import Arcface, l2_norm
from insightFace.recognition.arcface_torch.dataset import MXFaceDataset
import numpy as np


def main():

    device = 'cuda'

    num_classes = 93431
    model = get_model("mbf").to(device)  # MobileFaceNet
    head = Arcface(classnum=num_classes)
    classifier = None  # torch.nn.Linear(512, num_classes).to(device)
    optim = torch.optim.SGD(
        [{'params': model.parameters()}, {'params': head.parameters()}],
        lr=0.1, momentum=0.9, weight_decay=5e-4)
    dataset = MXFaceDataset(root_dir='/datasets/MS1M_RetinaFace_t1/ms1m-retinaface-t1', local_rank=0)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    # scheduler:
    decay_lr_milestones = torch.tensor([100000, 140000, 160000])  # iterations
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, decay_lr_milestones, gamma=0.1)

    max_iterations = 200000

    loss_fn = torch.nn.CrossEntropyLoss()

    tot_iter = 0
    epoch = 0
    print('Train Started!')
    start_time = time.time()
    while tot_iter < max_iterations:
        epoch += 1
        loss_prog, accuracy, tot_iter = \
            train_epoch(model, classifier, dataloader, optim, scheduler,
                        loss_fn, head, device, tot_iter, max_iterations)
        print(f'Epoch: {epoch} \t Iteration: {tot_iter} \t Mean Loss: {np.mean(loss_prog):.3} \t '
              f'Accuracy: {np.mean(accuracy)}')

    torch.save(model.state_dict(), 'mbf_dict.pt')
    torch.save(head.state_dict(), 'arcface_head_dict.pt')
    print(f'Train Finished! Took: {time.time() - start_time}')


def train_epoch(model, classifier, dataloader, optim, scheduler, loss_fn, head, device, tot_iter, max_iter):

    accuracy = []
    loss_progress = []
    for x, y in dataloader:
        start_time = time.time()
        optim.zero_grad()
        x, y = x.to(device), y.to(device)
        feats = l2_norm(model(x))
        # logits = classifier(feats)
        logits = head(feats, y)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        scheduler.step()
        loss_progress.append(loss.item())
        preds = logits.argmax(1)
        accuracy.append(torch.mean(preds.eq(y).float()).detach().cpu().numpy().item())
        tot_iter += 1
        if tot_iter % 10 == 0:
            print(f'Iteration: {tot_iter} \t Loss: {loss_progress[-1]:.3} \t Accuracy: {accuracy[-1]:.3} \t Avg iter Took {(time.time() - start_time)/10}')

        if tot_iter >= max_iter:
            break

    return loss_progress, accuracy, tot_iter



if __name__ == '__main__':
    main()

