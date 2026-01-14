from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import MYNET
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = self.get_base_dataloader_meta()
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
        return trainset, trainloader, testloader

    def get_base_dataloader_meta(self):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True)

        if self.args.dataset == 'cub200':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'StanfordCars':
            trainset = self.args.Dataset.StanfordCars(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.StanfordCars(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'StanfordDogs':
            trainset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'FGVCAircraft':
            trainset = self.args.Dataset.FGVCAircraft(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.FGVCAircraft(root=self.args.dataroot, train=False, index=class_index)
        # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.episode_way,
                                    self.args.episode_shot + self.args.episode_query)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=8,
                                                  pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.dataset == 'StanfordCars':
            trainset = self.args.Dataset.StanfordCars(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.dataset == 'StanfordDogs':
            trainset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)

        if self.args.dataset == 'FGVCAircraft':
            trainset = self.args.Dataset.FGVCAircraft(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=8, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=8, pin_memory=True)

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new)

        if self.args.dataset == 'StanfordCars':
            testset = self.args.Dataset.StanfordCars(root=self.args.dataroot, train=False,
                                                      index=class_new)

        if self.args.dataset == 'StanfordDogs':
            testset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=False,
                                                      index=class_new)
        if self.args.dataset == 'FGVCAircraft':
            testset = self.args.Dataset.FGVCAircraft(root=self.args.dataroot, train=False,
                                                     index=class_new)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp):
        for i in range(self.args.low_way):
            # random choose rotate degree
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)
            if sel_rot == 90:  # rotate 90 degree
                # print('rotate 90 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:  # rotate 180 degree
                # print('rotate 180 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
            elif sel_rot == 270:  # rotate 270 degree
                # print('rotate 270 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        return proto_tmp, query_tmp

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, args)

                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                    self.model.module.mode = 'avg_cos'

                    if args.set_no_val: # set no validation
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        tsl, tsa = self.test(self.model, testloader, args, session)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                    else:
                        # take the last session's testloader for validation
                        vl, va = self.validation()

                        # save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                          self.trlog['max_acc'][session]))
                        self.trlog['val_loss'].append(vl)
                        self.trlog['val_acc'].append(va)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)

                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                # always replace fc with avg mean
                self.model.load_state_dict(self.best_model_dict)
                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)

                embedding_list = []
                with torch.no_grad():
                    tqdm_gen = tqdm(testloader)
                    for _, batch in enumerate(tqdm_gen, 1):
                        data, _ = [_.cuda() for _ in batch]
                        embedding = self.model.module.encode(data)
                        embedding_list.append(embedding)

                embedding_list = torch.cat(embedding_list, dim=0)
                embedding_list = F.normalize(embedding_list, p=2, dim=-1)

                old_class = args.base_class + args.way * (session - 1)
                new_class = args.base_class + args.way * session

                proto = self.model.module.fc.weight.data.clone().detach()

                for i in range(new_class):
                    # print(new_class * self.num_trans)
                    # print(proto.shape)
                    vals = F.linear(F.normalize(proto[i, :].view(1, -1), p=2, dim=-1), embedding_list)
                    val, index = torch.topk(vals, k=args.R)
                    mask = val > args.t
                    index = index[:mask.sum()]
                    if embedding_list[index].shape[0] > 0:
                        # print(proto.shape)
                        # print(embedding_list[index].shape)
                        # proto[i, :] = (1 - args.base_calibration_degree) * proto[i, :] + args.base_calibration_degree * (embedding_list[index]).mean(1)
                        # 0
                        # proto[i, :] =  ((F.normalize(proto[i, :], p=2, dim=-1) + embedding_list[index]).sum(1) )/ (1+ embedding_list[index].shape[1])

                        # 1 final
                        proto[i, :] = (1 - args.base_calibration_degree) * F.normalize(proto[i, :], p=2, dim=-1) + args.base_calibration_degree * (embedding_list[index]).sum(1) / embedding_list[index].shape[1]

                self.model.module.fc.weight.data.copy_(proto)


                self.best_model_dict = deepcopy(self.model.state_dict())
                torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsa, inc_acc, harmonic_acc, correct_prediction_per_task, num_of_img_per_task, acc_list = self.test_t(self.model, testloader, args, session)

                self.trlog['correct_prediction_per_task'][session] = correct_prediction_per_task
                self.trlog['num_of_img_per_task'][session] = num_of_img_per_task
                self.trlog['acc_list'][session] = acc_list
                self.trlog['inc_acc'][session] = float('%.3f' % (inc_acc * 100))
                self.trlog['harmonic_acc'][session] = float('%.3f' % (harmonic_acc * 100))
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                print(correct_prediction_per_task)
                print(num_of_img_per_task)
                print(acc_list)

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f},\nbest test inc_acc {:.4f},\nbest test harm_acc {:.4f},\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], self.trlog['inc_acc'][session], self.trlog['harmonic_acc'][session] ))
                result_list.append(self.trlog['correct_prediction_per_task'][session])
                result_list.append(self.trlog['num_of_img_per_task'][session])
                result_list.append(self.trlog['acc_list'][session])


            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                embedding_list = []
                with torch.no_grad():
                    tqdm_gen = tqdm(testloader)
                    for _, batch in enumerate(tqdm_gen, 1):
                        data, _ = [_.cuda() for _ in batch]
                        embedding = self.model.module.encode(data)
                        embedding_list.append(embedding)

                embedding_list = torch.cat(embedding_list, dim=0)
                embedding_list = F.normalize(embedding_list, p=2, dim=-1)

                old_class = args.base_class + args.way * (session - 1)
                new_class = args.base_class + args.way * session

                proto = self.model.module.fc.weight.data.clone().detach()

                for i in range(new_class):
                    vals = F.linear(F.normalize(proto[i, :].view(1, -1), p=2, dim=-1), embedding_list)
                    val, index = torch.topk(vals, k=args.R)
                    mask = val > args.t
                    index = index[:mask.sum()]
                    if embedding_list[index].shape[0] > 0:

                        if i < args.base_class:
                            # 0
                            # proto[i, :] = (F.normalize(proto[i, :], p=2, dim=-1) + (embedding_list[index]).sum(1))/ (1+ embedding_list[index].shape[1])

                            # # 1
                            # proto[i, :] = (1 - args.base_calibration_degree) * F.normalize(proto[i, :], p=2, dim=-1) + (
                            #                       args.base_calibration_degree) * (
                            #           embedding_list[index]).sum(1) / embedding_list[index].shape[1]

                            # # final
                            proto[i, :] = (1 - args.base_calibration_degree ** (session + 1)) * F.normalize(proto[i, :], p=2, dim=-1) + (args.base_calibration_degree ** (session + 1)) * (embedding_list[index]).sum(1) / embedding_list[index].shape[1]
                        else:
                            # 0
                            # proto[i, :] = (F.normalize(proto[i, :], p=2, dim=-1) + (embedding_list[index]).sum(1)) / (
                            #         1 + embedding_list[index].shape[1])

                            # # 1
                            # proto[i, :] = (1 - args.base_calibration_degree) * F.normalize(proto[i, :], p=2, dim=-1) + (args.base_calibration_degree) * (
                            #           embedding_list[index]).sum(1) / embedding_list[index].shape[1]

                            # # final
                            weight = args.inc_calibration_degree ** (
                                    session - (i - args.base_class ) // args.way)
                            proto[i, :] = (1 - weight) * F.normalize(proto[i, :], p=2, dim=-1) + weight * (
                                embedding_list[index]).sum(1) / embedding_list[index].shape[1]

                self.model.module.fc.weight.data.copy_(proto)


                tsa, inc_acc, harmonic_acc, correct_prediction_per_task, num_of_img_per_task, acc_list = self.test_t(self.model, testloader, args, session)

                # save better model
                self.trlog['correct_prediction_per_task'][session] = correct_prediction_per_task
                self.trlog['num_of_img_per_task'][session] = num_of_img_per_task
                self.trlog['acc_list'][session] = acc_list
                self.trlog['inc_acc'][session] = float('%.3f' % (inc_acc * 100))
                self.trlog['harmonic_acc'][session] = float('%.3f' % (harmonic_acc * 100))
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))



                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                print(correct_prediction_per_task)
                print(num_of_img_per_task)
                print(acc_list)




                result_list.append(
                    'Session {}, Test Best Epoch {},\nbest test Acc {:.4f},\nbest test inc_acc {:.4f},\nbest test harm_acc {:.4f},\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session],
                        self.trlog['inc_acc'][session], self.trlog['harmonic_acc'][session]))
                result_list.append( self.trlog['correct_prediction_per_task'][session])
                result_list.append( self.trlog['num_of_img_per_task'][session])
                result_list.append( self.trlog['acc_list'][session])

        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, testloader, self.args, session)

        return vl, va

    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()

        tqdm_gen = tqdm(trainloader)

        label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for i, batch in enumerate(tqdm_gen, 1):
            data, true_label = [_.cuda() for _ in batch]

            k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]
            # sample low_way data
            proto_tmp = deepcopy(
                proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
                :args.low_shot,
                :args.low_way, :, :, :].flatten(0, 1))
            query_tmp = deepcopy(
                query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
                :args.low_way, :, :, :].flatten(0, 1))
            # random choose rotate degree
            proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)

            model.module.mode = 'encoder'
            data = model(data)
            proto_tmp = model(proto_tmp)
            query_tmp = model(query_tmp)

            # k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]

            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
            query = query.view(args.episode_query, args.episode_way, query.shape[-1])

            proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
            query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

            proto = proto.mean(0).unsqueeze(0)
            proto_tmp = proto_tmp.mean(0).unsqueeze(0)

            proto = torch.cat([proto, proto_tmp], dim=1)
            query = torch.cat([query, query_tmp], dim=1)

            proto = proto.unsqueeze(0)
            query = query.unsqueeze(0)

            logits = model.module._forward(proto, query)

            total_loss = F.cross_entropy(logits, label)

            acc = count_acc(logits, label)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def test(self, model, testloader, args, session):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits = model.module._forward(proto, query)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

        return vl, va


    def lp_distance(self, x, y, p=3):
        return np.sum(np.abs(x-y)**p)**(1/p)

    def test_t(self, model, testloader, args, session):
        model = model.eval()
        test_class = args.base_class + session * args.way
        embedding_list = []
        label_list = []
        # proto_list = []

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                embedding = model(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                # proto_list.append(proto.cpu())
                # proto = proto.unsqueeze(0).unsqueeze(0)

                # logits = model.module._forward(proto, query)
                #
                # loss = F.cross_entropy(logits, test_label)
                # acc = count_acc(logits, test_label)

        embedding_list = torch.cat(embedding_list, dim=0).cpu()
        # embedding_list1 = torch.nn.functional.normalize(embedding_list, p=2, dim=-1)

        proto_list = model.module.fc.weight[:test_class, :].detach().cpu()

        # proto_list1 = torch.nn.functional.normalize(proto_list, p=2, dim=-1)

        label_list = torch.cat(label_list, dim=0).cpu()

        # for i in range(len(cls_wise_feature_prototype)):
        #     cls_wise_feature_prototype[i] = cls_wise_feature_prototype[i].view(-1)
        # proto_list = torch.stack(cls_wise_feature_prototype, dim=0).cpu()
        # proto_list = torch.nn.functional.normalize(proto_list, p=2, dim=-1)

        # metric: euclidean, cosine, l1, l2, l3
        pairwise_distance = pairwise_distances(np.asarray(embedding_list), np.asarray(proto_list), metric='cosine')
        # pairwise_distance = -np.asarray(F.linear(F.normalize(embedding_list, p=2, dim=-1), F.normalize(proto_list, p=2, dim=-1)).cpu())
        pairwise_distance = pairwise_distance/np.mean(pairwise_distance)
        prediction_result = np.argmin(pairwise_distance, axis=1)

        pairwise_distance1 = pairwise_distances(np.asarray(embedding_list), np.asarray(proto_list), metric='l1')
        pairwise_distance1 = pairwise_distance1/np.mean(pairwise_distance1)
        prediction_result1 = np.argmin(pairwise_distance1, axis=1)

        # pairwise_distance11 = pairwise_distances(np.asarray(embedding_list1), np.asarray(proto_list1), metric='l1')
        # prediction_result11 = np.argmin(pairwise_distance11, axis=1)

        pairwise_distance2 = pairwise_distances(np.asarray(embedding_list), np.asarray(proto_list), metric='l2')
        pairwise_distance2 = pairwise_distance2/np.mean(pairwise_distance2)
        prediction_result2 = np.argmin(pairwise_distance2, axis=1)

        # pairwise_distance3 = pairwise_distances(np.asarray(embedding_list), np.asarray(proto_list), metric=self.lp_distance, p = 3)
        pairwise_distance3 = np.asarray(torch.cdist(embedding_list.cuda(), proto_list.cuda(), p=3).cpu())
        pairwise_distance3 = pairwise_distance3 / np.mean(pairwise_distance3)
        prediction_result3 = np.argmin(pairwise_distance3, axis=1)

        # print(pairwise_distance, pairwise_distance1, pairwise_distance2, pairwise_distance3)




        label_list = np.asarray(label_list)
        total_acc = np.sum(prediction_result == label_list) / float(len(label_list))
        total_acc1 = np.sum(prediction_result1 == label_list) / float(len(label_list))
        # total_acc11 = np.sum(prediction_result11 == label_list) / float(len(label_list))
        total_acc2 = np.sum(prediction_result2 == label_list) / float(len(label_list))
        total_acc3 = np.sum(prediction_result3 == label_list) / float(len(label_list))





        num_of_img_per_task = [0] * args.sessions
        correct_prediction_per_task = [0] * args.sessions
        acc_list = [0.0] * args.sessions
        avg_inc = 0
        avg_pinc = 0
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
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_inc = avg_inc + acc_list[i]
                if i != 0 and i!=session:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_pinc = avg_pinc + acc_list[i]
        avg_inc = avg_inc / session if session > 0 else 1
        avg_pinc = avg_pinc / (session-1) if session > 1 else 1
        print('COS', acc_list, avg_inc, avg_pinc, total_acc)



        num_of_img_per_task1 = [0] * args.sessions
        correct_prediction_per_task1 = [0] * args.sessions
        acc_list1 = [0.0] * args.sessions
        avg_inc1 = 0
        avg_pinc1 = 0

        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way

            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task1[i] = num_of_img_per_task1[i] + 1
                    if label_list[k] == prediction_result1[k]:
                        correct_prediction_per_task1[i] = correct_prediction_per_task1[i] + 1

            if num_of_img_per_task1[i] != 0:
                acc_list1[i] = correct_prediction_per_task1[i] / num_of_img_per_task1[i]
                if i != 0:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_inc1 = avg_inc1 + acc_list1[i]
                if i != 0 and i != session:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_pinc1 = avg_pinc1 + acc_list1[i]
        avg_inc1 = avg_inc1 / session if session > 0 else 1
        avg_pinc1 = avg_pinc1 / (session - 1) if session > 1 else 1
        print('L1', acc_list1, avg_inc1, avg_pinc1, total_acc1)



        num_of_img_per_task2= [0] * args.sessions
        correct_prediction_per_task2= [0] * args.sessions
        acc_list2 = [0.0] * args.sessions
        avg_inc2 = 0
        avg_pinc2 = 0
        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way

            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task2[i] = num_of_img_per_task2[i] + 1
                    if label_list[k] == prediction_result2[k]:
                        correct_prediction_per_task2[i] = correct_prediction_per_task2[i] + 1

            if num_of_img_per_task2[i] != 0:
                acc_list2[i] = correct_prediction_per_task2[i] / num_of_img_per_task2[i]
                if i != 0:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_inc2 = avg_inc2 + acc_list2[i]
                if i != 0 and i != session:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_pinc2 = avg_pinc2 + acc_list2[i]
        avg_inc2 = avg_inc2 / session if session > 0 else 1
        avg_pinc2 = avg_pinc2 / (session - 1) if session > 1 else 1
        print('L2', acc_list2, avg_inc2, avg_pinc2, total_acc2)

        # BL1 = torch.norm(embedding_list[:num_of_img_per_task[0]], dim=-1, p=1).mean()
        # IL1 = torch.norm(embedding_list[num_of_img_per_task[0]:], dim=-1, p=1).mean()
        #
        # (BL1-IL1)/IL1


        pairwise_distance12 = (pairwise_distance1 + pairwise_distance2) / 2
        prediction_result12 = np.argmin(pairwise_distance12, axis=1)

        pairwise_distance123 = (pairwise_distance1 + pairwise_distance2 + pairwise_distance3) / 3
        prediction_result123 = np.argmin(pairwise_distance123, axis=1)

        pairwise_distance01 = (1 - args.LW) * pairwise_distance + args.LW * pairwise_distance1
        prediction_result01 = np.argmin(pairwise_distance01, axis=1)

        pairwise_distance01 = (1 - args.LW) * pairwise_distance + args.LW * pairwise_distance1
        prediction_result01 = np.argmin(pairwise_distance01, axis=1)

        pairwise_distance02 = (1 - args.LW) * pairwise_distance + args.LW * pairwise_distance2
        prediction_result02 = np.argmin(pairwise_distance02, axis=1)

        pairwise_distance012 = (1 - args.LW) * pairwise_distance + args.LW * (
                pairwise_distance1 + pairwise_distance2) / 2
        prediction_result012 = np.argmin(pairwise_distance012, axis=1)

        pairwise_distance0123 = (1 - args.LW) * pairwise_distance + args.LW * (
                pairwise_distance1 + pairwise_distance2 + pairwise_distance3) / 3
        prediction_result0123 = np.argmin(pairwise_distance0123, axis=1)

        if session > 0:
            prediction_result20 = np.where(prediction_result >= args.base_class, np.argmin(pairwise_distance2, axis=1),
                                           prediction_result)
        else:
            prediction_result20 = prediction_result

        total_acc12 = np.sum(prediction_result12 == label_list) / float(len(label_list))
        total_acc123 = np.sum(prediction_result123 == label_list) / float(len(label_list))
        total_acc01 = np.sum(prediction_result01 == label_list) / float(len(label_list))
        total_acc02 = np.sum(prediction_result02 == label_list) / float(len(label_list))
        total_acc012 = np.sum(prediction_result012 == label_list) / float(len(label_list))
        total_acc0123 = np.sum(prediction_result0123 == label_list) / float(len(label_list))
        total_acc20 = np.sum(prediction_result20 == label_list) / float(len(label_list))






        num_of_img_per_task012 = [0] * args.sessions
        correct_prediction_per_task012 = [0] * args.sessions
        acc_list012 = [0.0] * args.sessions
        avg_inc012 = 0
        avg_pinc012 = 0
        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way

            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task012[i] = num_of_img_per_task012[i] + 1
                    if label_list[k] == prediction_result012[k]:
                        correct_prediction_per_task012[i] = correct_prediction_per_task012[i] + 1

            if num_of_img_per_task012[i] != 0:
                acc_list012[i] = correct_prediction_per_task012[i] / num_of_img_per_task012[i]
                if i != 0:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_inc012 = avg_inc012 + acc_list012[i]
                if i != 0 and i != session:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_pinc012 = avg_pinc012 + acc_list012[i]
        avg_inc012 = avg_inc012 / session if session > 0 else 1
        avg_pinc012 = avg_pinc012 / (session - 1) if session > 1 else 1

        print('COS+L1+L2', acc_list012, avg_inc012, avg_pinc012, total_acc012)



        num_of_img_per_task01 = [0] * args.sessions
        correct_prediction_per_task01 = [0] * args.sessions
        acc_list01 = [0.0] * args.sessions
        avg_inc01 = 0
        avg_pinc01 = 0

        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way

            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task01[i] = num_of_img_per_task01[i] + 1
                    if label_list[k] == prediction_result01[k]:
                        correct_prediction_per_task01[i] = correct_prediction_per_task01[i] + 1

            if num_of_img_per_task01[i] != 0:
                acc_list01[i] = correct_prediction_per_task01[i] / num_of_img_per_task01[i]
                if i != 0:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_inc01 = avg_inc01 + acc_list01[i]
                if i != 0 and i != session:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_pinc01 = avg_pinc01 + acc_list01[i]
        avg_inc01 = avg_inc01 / session if session > 0 else 1
        avg_pinc01 = avg_pinc01 / (session - 1) if session > 1 else 1

        print('COS+L1', acc_list01, avg_inc01, avg_pinc01, total_acc01)



        num_of_img_per_task02 = [0] * args.sessions
        correct_prediction_per_task02 = [0] * args.sessions
        acc_list02 = [0.0] * args.sessions
        avg_inc02 = 0
        avg_pinc02 = 0

        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way

            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task02[i] = num_of_img_per_task02[i] + 1
                    if label_list[k] == prediction_result02[k]:
                        correct_prediction_per_task02[i] = correct_prediction_per_task02[i] + 1

            if num_of_img_per_task02[i] != 0:
                acc_list02[i] = correct_prediction_per_task02[i] / num_of_img_per_task02[i]
                if i != 0:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_inc02 = avg_inc02 + acc_list02[i]
                if i != 0 and i != session:
                    # square = square + torch.square(acc_list[i]-total_acc)
                    avg_pinc02 = avg_pinc02 + acc_list02[i]
        avg_inc02 = avg_inc02 / session if session > 0 else 1
        avg_pinc02 = avg_pinc02 / (session - 1) if session > 1 else 1

        print('COS+L2', acc_list02, avg_inc02, avg_pinc02, total_acc02)


        print('BL3', torch.norm(embedding_list[:num_of_img_per_task[0]], dim=-1, p=3).mean())
        print('BL2', torch.norm(embedding_list[:num_of_img_per_task[0]], dim=-1).mean())
        print('BL1', torch.norm(embedding_list[:num_of_img_per_task[0]], dim=-1, p=1).mean())

        print('IL3', torch.norm(embedding_list[num_of_img_per_task[0]:], dim=-1, p=3).mean())
        print('IL2', torch.norm(embedding_list[num_of_img_per_task[0]:], dim=-1).mean())
        print('IL1', torch.norm(embedding_list[num_of_img_per_task[0]:], dim=-1, p=1).mean())

        print('PIL3', torch.norm(embedding_list[num_of_img_per_task[0] :-num_of_img_per_task[session] ], dim=-1, p=3).mean())
        print('PIL2', torch.norm(embedding_list[num_of_img_per_task[0] :-num_of_img_per_task[session] ], dim=-1).mean())
        print('PIL1', torch.norm(embedding_list[num_of_img_per_task[0] :-num_of_img_per_task[session] ], dim=-1, p=1).mean())

        print('CIL3', torch.norm(embedding_list[-num_of_img_per_task[session] :], dim=-1, p=3).mean())
        print('CIL2', torch.norm(embedding_list[-num_of_img_per_task[session] :], dim=-1).mean())
        print('CIL1', torch.norm(embedding_list[-num_of_img_per_task[session] :], dim=-1, p=1).mean())





        harmonic_acc = (2 * acc_list[0] * avg_inc) / (np.square(acc_list[0]) + np.square(avg_inc))
        # harmonic_acc = np.square(acc_list[0]-total_acc) + square_inc
        # harmonic_acc = acc_list[0] - avg_inc

        return total_acc, avg_inc, harmonic_acc, correct_prediction_per_task, num_of_img_per_task, acc_list



    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + '%dW-%dS-%dQ-%dEpi-L%dW-L%dS' % (
            self.args.episode_way, self.args.episode_shot, self.args.episode_query, self.args.train_episode,
            self.args.low_way, self.args.low_shot)
        # if self.args.use_euclidean:
        #     self.args.save_path = self.args.save_path + '_L2/'
        # else:
        #     self.args.save_path = self.args.save_path + '_cos/'
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Lrg_%.5f-MS_%s-Gam_%.2f-T_%.2f-lw_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.lrg, mile_stone, self.args.gamma,
                self.args.temperature, self.args.LW)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Lrg_%.5f-Step_%d-Gam_%.2f-T_%.2f-lw_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.lrg, self.args.step, self.args.gamma,
                self.args.temperature, self.args.LW)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
