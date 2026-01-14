from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)

                    embedding_list = []
                    label_list = []

                    with torch.no_grad():
                        tqdm_gen = tqdm(testloader)
                        for _, batch in enumerate(tqdm_gen, 1):
                            data, label = [_.cuda() for _ in batch]
                            embedding = self.model.module.encode(data)
                            embedding_list.append(embedding)
                            label_list.append(label)

                    embedding_list = torch.cat(embedding_list, dim=0)
                    embedding_list = F.normalize(embedding_list, p=2, dim=-1)
                    label_list = torch.cat(label_list, dim=0)

                    new_class = args.base_class + args.way * session

                    proto = self.model.module.fc.weight.data.clone().detach()

                    for i in range(new_class):

                        vals = F.linear(F.normalize(proto[i, :].view(1, -1), p=2, dim=-1), embedding_list)
                        val, index = torch.topk(vals, k=args.R)
                        mask = val > args.t
                        index = index[:mask.sum()]
                        if embedding_list[index].shape[0] > 0:

                            proto[i, :] = (1 - args.base_calibration_degree) * F.normalize(proto[i, :], p=2,
                                                                                           dim=-1) + args.base_calibration_degree * (
                                              embedding_list[index]).sum(1) / embedding_list[index].shape[1]

                    self.model.module.fc.weight.data.copy_(proto)


                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsa, inc_acc = self.test_t(
                        self.model, embedding_list, label_list, args, session)

                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                embedding_list = []
                label_list = []

                with torch.no_grad():
                    tqdm_gen = tqdm(testloader)
                    for _, batch in enumerate(tqdm_gen, 1):
                        data, label = [_.cuda() for _ in batch]
                        embedding = self.model.module.encode(data)
                        embedding_list.append(embedding)
                        label_list.append(label)

                embedding_list = torch.cat(embedding_list, dim=0)
                embedding_list = F.normalize(embedding_list, p=2, dim=-1)
                label_list = torch.cat(label_list, dim=0)

                new_class = args.base_class + args.way * session

                proto = self.model.module.fc.weight.data.clone().detach()

                for i in range(new_class):
                    vals = F.linear(F.normalize(proto[i, :].view(1, -1), p=2, dim=-1), embedding_list)
                    val, index = torch.topk(vals, k=args.R)
                    mask = val > args.t
                    index = index[:mask.sum()]
                    if embedding_list[index].shape[0] > 0:

                        if i < args.base_class:

                            proto[i, :] = (1 - args.base_calibration_degree ** (session + 1)) * F.normalize(proto[i, :],
                                                                                                            p=2,
                                                                                                            dim=-1) + (
                                                  args.base_calibration_degree ** (session + 1)) * (
                                              embedding_list[index]).sum(1) / embedding_list[index].shape[1]
                        else:

                            weight = args.inc_calibration_degree ** (
                                    session - (i - args.base_class) // args.way)
                            proto[i, :] = (1 - weight) * F.normalize(proto[i, :], p=2, dim=-1) + weight * (
                                embedding_list[index]).sum(1) / embedding_list[index].shape[1]

                self.model.module.fc.weight.data.copy_(proto)


                tsa, inc_acc = self.test_t(self.model, embedding_list, label_list, args, session)


                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def test_t(self, model, embedding_list, label_list, args, session):
        model = model.eval()
        test_class = args.base_class + session * args.way

        embedding_list = embedding_list.cpu()
        proto_list = model.module.fc.weight[:test_class, :].detach().cpu()

        label_list = label_list.cpu()

      
        pairwise_distance = pairwise_distances(np.asarray(embedding_list), np.asarray(proto_list), metric='cosine')
        pairwise_distance = pairwise_distance/np.mean(pairwise_distance)
        prediction_result = np.argmin(pairwise_distance, axis=1)
  
        label_list = np.asarray(label_list)
        total_acc = np.sum(prediction_result == label_list) / float(len(label_list))
      
        num_of_img_per_task = [0] * args.sessions
        correct_prediction_per_task = [0] * args.sessions
        acc_list = [0.0] * args.sessions
        avg_inc = 0
        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way

            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task[i] = num_of_img_per_task[i] + 1
                    if label_list[k] == prediction_result[k]:
                        correct_prediction_per_task[i] = correct_prediction_per_task[i] + 1

            if num_of_img_per_task[i] != 0:
                acc_list[i] = correct_prediction_per_task[i] / num_of_img_per_task[i]
                if i != 0:
                    avg_inc = avg_inc + acc_list[i]
        avg_inc = avg_inc / session if session > 0 else 1
        print('Overall acc:', total_acc, 'Incremental acc:', avg_inc)
        return total_acc, avg_inc
    

   


    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f-LW_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.LW)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f-LW_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.LW)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
