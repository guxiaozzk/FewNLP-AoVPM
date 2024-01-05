import copy
import os
import torch
import time
import sklearn.metrics as metrics
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from param import args
from utils import processors
from modelforatt import MSAModel
from dataset2 import MSADataset
from transformers.optimization import AdamW
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np


if args.dataset in ['t2015', 't2017', 'masad']:
    print('[#] Aspect-level')
else:
    print('[#] Sentence-level')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # load datasets and create dataloaders
        self.train_set = MSADataset(args, processors[args.dataset], mode='train', max_seq_length=64 if args.dataset=='masad' else 128)
        self.train_loader = DataLoader(self.train_set, collate_fn=self.train_set.collate_fn, batch_size=args.batch_size, shuffle=True, drop_last=False)
        self.dev_set = MSADataset(args, processors[args.dataset], mode='dev', max_seq_length=64 if args.dataset=='masad' else 128)
        self.dev_loader = DataLoader(self.dev_set, collate_fn=self.dev_set.collate_fn, batch_size=args.batch_size * 4)
        self.test_set = MSADataset(args, processors[args.dataset], mode='test', max_seq_length=64 if args.dataset=='masad' else 128)
        self.test_loader = DataLoader(self.test_set, collate_fn=self.test_set.collate_fn, batch_size=args.batch_size * 4)
        # self.test_loader = DataLoader(self.test_set, collate_fn=self.test_set.collate_fn, batch_size=408)
        # create model
        self.model = MSAModel(args, self.test_set.label_id_list)
        self.model.to(self.device)

        # create ckpt
        self.best_ckpt = {
            'test_size': len(self.test_set),
            'args': self.args
        }

        self.save_dir = self.get_save_dir()
        os.makedirs(self.save_dir, exist_ok=True)

    def get_save_dir(self):
        if '18' in self.args.few_shot_file:
            sample = '[18]'
        elif '36' in self.args.few_shot_file:
            sample = '[36]'
        elif '72' in self.args.few_shot_file:
            sample = '[72]'
        elif '108' in self.args.few_shot_file:
            sample = '[108]'
        elif '144' in self.args.few_shot_file:
            sample = '[144]'
        elif '1' in self.args.few_shot_file:
            sample = '[s1]'
        elif '2' in self.args.few_shot_file:
            sample = '[s2]'
        else:
            sample = '[s0]'
        template_name = '{}[t{}]'.format(sample, self.args.template)
        if self.args.template == 3:
            template_name += '[{}]'.format(self.args.prompt_shape)
        if not self.args.no_img:
            template_name += "[{}-{}]".format(self.args.visual_model_name, self.args.img_token_len)
        
        if self.args.lr_visual_encoder and self.args.lr_visual_encoder > 0:
            return os.path.join(self.args.out_dir, self.args.dataset, template_name, 'lrv_{}'.format(self.args.lr_visual_encoder), str(self.args.lr_lm_model))
        else:
            return os.path.join(self.args.out_dir, self.args.dataset, template_name, str(self.args.lr_lm_model))

    def save(self):
        '''save predictions but
        do not save model parameters
        '''
        ckpt_name = self.best_ckpt['ckpt_name']
        test_y = self.best_ckpt['test_y']
        test_y_ = self.best_ckpt['test_y_']
        with open(os.path.join(self.save_dir, ckpt_name) + '.txt', 'w', encoding='utf-8') as f:
            f.write('#True\t#Pred\n')
            for y, y_ in zip(test_y, test_y_):
                token_y = self.test_set.tokenizer.convert_ids_to_tokens(y)
                token_y_ = self.test_set.tokenizer.convert_ids_to_tokens(y_)
                f.write(f'{token_y:10}\t{token_y_:10}\n')
        print("[#] Checkpoint {} saved.".format(ckpt_name))
        return

    def load(self):
        """load ckpt
        """
        if self.args.ckpt_name == None:
            print("[#] Loading nothing, using BERT params")
            return
        ckpt_dict = torch.load(self.args.ckpt_name, map_location='cpu')
        print(f'[#] Loading {ckpt_dict["ckpt_name"]}')
        print(f'[#] dev_acc {ckpt_dict["dev_acc"]}')
        print(f'[#] test_acc {ckpt_dict["test_acc"]}')
        print(f'[#] test_mac_f1 {ckpt_dict["test_mac_f1"]}')
        self.model.load_state_dict(ckpt_dict["embedding"])

    def _evaluate(self, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
        else:
            loader = self.dev_loader
        with torch.no_grad():
            self.model.eval()
            loss = []
            y_ = []
            y = []
            probs_merge = []
            pbar = tqdm(loader, unit="batch", desc=f'*{evaluate_type} pbar')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
                
                _loss, _y_, _y,p_rep,img_rep,probs = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)
                probs_merge.extend(probs)
#                 if evaluate_type == 'Test':
#                     # print(_y)
                    
#                     with open('represent.txt','w') as f:
#                         print("save")
#                         for it in list(p_rep):
                            
#                             f.writelines(str(it))
#                     with open('imgrepresent.txt','w') as f2:
#                         print("save")
#                         for it in list(img_rep):
#             #                 print(it)
                            
#                             f2.writelines(str(it))
#                     # print(li)

            
            loss = sum(loss) / len(loader)
            print(len(y),len(y_))
            acc = metrics.accuracy_score(y, y_)
            f1_macro = metrics.f1_score(y, y_, average='macro')
            f1_weighted  = metrics.f1_score(y, y_, average='weighted')
            r1_macro = metrics.recall_score(y, y_, average='macro')
            r1_weighted  = metrics.recall_score(y, y_, average='weighted')

            print(f"[{evaluate_type:5}] Loss: {loss:0.4f} Acc: {acc:0.4f}, Macro F1: {f1_macro:0.4f}  weight F1: {f1_weighted:0.4f},Macro r1: {r1_macro:0.4f}  weight r1: {r1_weighted:0.4f} ")
            
            # print(f"[{evaluate_type:5}] Loss: {loss:0.4f} Acc: {acc:0.4f}, Macro F1: {f1_macro:0.4f}  weight F1: {f1_weighted:0.4f}")
        return loss, acc, f1_weighted, y, y_,probs_merge

    def update_best_ckpt(self, epoch_idx=None, dev_acc=None, test_acc=None, test_f1=None, test_y=None, test_y_=None):
        if test_acc == None:  # during training
            model_params = copy.deepcopy(self.model.state_dict())
            self.best_ckpt['time'] = datetime.now()
            self.best_ckpt['embedding'] = model_params
            self.best_ckpt['epoch'] = epoch_idx
            self.best_ckpt['dev_acc'] = dev_acc
        else:  # after testing
            ckpt_name = time.strftime("%y%m%d_%H:%M:%S", time.localtime())
            epoch = epoch_idx if epoch_idx else self.best_ckpt['epoch']
            ckpt_name += "[Ep{}][Test{}-{}][Dev{}].ckpt".format(epoch, round(test_acc * 100, 2), round(test_f1 * 100, 2), round(self.best_ckpt['dev_acc'] * 100, 2))
            self.best_ckpt['ckpt_name'] = ckpt_name
            # self.best_ckpt['dev_acc'] = dev_acc
            self.best_ckpt['test_acc'] = test_acc
            self.best_ckpt['test_mac_f1'] = test_f1
            self.best_ckpt['test_y'] = test_y
            self.best_ckpt['test_y_'] = test_y_
        
    def train(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        best_dev_acc, early_stop, best_dev_f1,best_dev_loss= 0, 0,0,0
        
        params = []
        params.append({'params': self.model.lm_model.parameters(), 'lr': self.args.lr_lm_model})
        if self.args.template == 3:
            params.append({'params': self.model.prompt_encoder.parameters(), 'lr': self.args.lr_lm_model})
        if not self.args.no_img:
            params.append({'params': self.model.visual_encoder.backbone.parameters(), 'lr': self.args.lr_visual_encoder})
            params.append({'params': self.model.visual_encoder.visual_mlp.parameters(), 'lr': self.args.lr_lm_model})
        
        optimizer = AdamW(params=params, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(100):
            print(f'\n[#] Epoch {epoch_idx}')
            loss = []
            y_ = []
            y = []

            pbar = tqdm(self.train_loader, unit="batch")
            for batch in pbar:
                self.model.train()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
            
                _loss, _y_, _y,p_rep,img_rep,probs = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)
                pbar.set_description(f'*Train batch loss: {_loss.item():0.4f}')

                _loss.backward()
                # torch.cuda.empty_cache()
                optimizer.step()
                # torch.cuda.empty_cache()
                optimizer.zero_grad()
                   # print('p_rep,img_rep',p_rep.shape,img_rep.shape)
#                 with open('represent.txt','w') as f:

#                     for it in list(p_rep):
#         #                 print(it)
#                         f.writelines(str(it))
#                 with open('imgrepresent.txt','w') as f2:

#                     for it in list(img_rep):
#         #                 print(it)
#                         f2.writelines(str(it))
            my_lr_scheduler.step()
            loss = sum(loss) / len(self.train_loader)
            print(len(y),len(y_))
            
            acc = metrics.accuracy_score(y, y_)
          
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print("本批用时",start.elapsed_time(end))

            print(f"[Train] Loss: {loss:0.4f} Hit@1: {acc:0.4f}")
            print('learning_rate',optimizer.state_dict()['param_groups'][0]['lr'])
            dev_loss, dev_acc, dev_f1, _, _,probs = self._evaluate('Dev')
            _, test_acc, test_f1, test_y, test_y_,probs = self._evaluate('Test')
            if dev_f1 >= best_dev_f1:
                print(f'[#] Best dev acc: {dev_acc:0.4f} Best weight f1: {dev_f1:0.4f}')
               
                print(early_stop)
#                 print("save reps")
            
#                 print("保存进去的参数 ",self.model.state_dict())
#                 for param_tensor in self.model.state_dict():
#                     #打印 key value字典
#                     print(param_tensor,'\t',self.model.state_dict()[param_tensor].size())
                dic_o  = self.model.state_dict()
                self.update_best_ckpt(epoch_idx, dev_acc)
                                
                early_stop = 0
                best_dev_acc = dev_acc
                best_dev_f1 = dev_f1
                
                # self.model.load_state_dict(self.best_ckpt["embedding"])
#                 print("加载出来的参数 ",self.model.state_dict())
#                 for param_tensor in self.model.state_dict():
#                     #打印 key value字典
#                     print(param_tensor,'\t',self.model.state_dict()[param_tensor].size())
#                 print()
#                 dict_diff =  cmpdicts(dic_o,self.model.state_dict())
#                 print ("字典不同的值")
#                 print (dict_diff)
#                 print(dic_o.keys() & self.model.state_dict().keys())
                # _, test_acc, test_f1_mac, test_y, test_y_ = self._evaluate('Test')
                import sklearn.metrics as sm
                from sklearn.metrics import roc_curve, auc
                import seaborn as sn
                import itertools
                import matplotlib.pyplot as plt

                import  numpy as np
                def Find_Optimal_Cutoff(TPR, FPR, threshold):
                    y = TPR - FPR
                    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
                    optimal_threshold = threshold[Youden_index]
                    point = [FPR[Youden_index], TPR[Youden_index]]
                    return optimal_threshold, point
                def ROC(label, y_prob):
                    """
                    Receiver_Operating_Characteristic, ROC
                    :param label: (n, )
                    :param y_prob: (n, )
                    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
                    """
                    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
                    roc_auc = metrics.auc(fpr, tpr)
                    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
                    return fpr, tpr, roc_auc, optimal_th, optimal_point

#                 test_pred ,y_preob= test_y_,test_y
#                 pred_1,pred_2,pred_3,y_1,y_2,y_3 = np.zeros(len(test_pred)),np.zeros(len(test_pred)),np.zeros(len(test_pred)),np.zeros(len(test_pred)),np.zeros(len(test_pred)),np.zeros(len(test_pred))

#                 def get_biclass(pred_1,y_1,test_pred,y_predob,label):
#                     for i in range(len(test_pred)):
#                         if test_pred[i]==label:
#                             pred_1[i]=1
#                         else:
#                             pred_1[i]=0
#                         if y_predob[i]==label:
#                             y_1[i]=1
#                         else:
#                             y_1[i]=0
#                     return pred_1,y_1
# #                 # return pred_1,y_1
#                 pred_1,y_1 = get_biclass(pred_1,y_1,test_pred,y_preob,0)
#                 pred_2,y_2 = get_biclass(pred_2,y_2,test_pred,y_preob,1)
#                 pred_3,y_3 = get_biclass(pred_3,y_3,test_pred,y_preob,2)
#                 # print(pred_1,y_1)
#                 fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(pred_1, y_1)
#                 fpr2, tpr2, roc_auc2, optimal_th_2, optimal_point_2 = ROC(pred_2, y_2)
#                 fpr3, tpr3, roc_auc3, optimal_th_3, optimal_point_3 = ROC(pred_3, y_3)

#                 # pred_1,y_1 = get_biclass(pred_1,y_1,test_pred,y_preob,2)
#                 # print(pred_1,y_1)
#                 # for i in range(n_classes):
#                 #     mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

#                 # Average it and compute AUC
#                 fpr_grid = np.linspace(0.0, 1.0, 1000)

#                 # Interpolate all ROC curves at these points
#                 mean_tpr = np.zeros_like(fpr_grid)

#                 mean_tpr+= np.interp(fpr_grid, fpr,tpr)  # linear interpolation

#                 mean_tpr+= np.interp(fpr_grid, fpr2,tpr2)  # linear interpolation

#                 mean_tpr+= np.interp(fpr_grid, fpr3,tpr3)  # linear interpolation

#                 mean_tpr /= 3

#                 fpr4 = fpr_grid
#                 tpr4 = mean_tpr
#                 roc_auc4= auc(fpr4, tpr4)

#                 print(f"subolot loc1 Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc4:.2f}")
#                 fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(pred_1, y_1)

#                 fig, ax = plt.subplots() # 创建图实例

#                 ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
#                 ax.plot([0, 1], [0, 1], linestyle="--")
#                 ax.plot(fpr2, tpr2, label=f"AUC = {roc_auc2:.3f}")
#                 ax.plot(fpr3, tpr3, label=f"AUC = {roc_auc3:.3f}")
#                 ax.plot(fpr4, tpr4, label=f"AUC = {roc_auc4:.3f}")

#                 # ax.plot(optimal_point[0], optimal_point[1],label="l1")
#                 # ax.plot(optimal_point_2[0], optimal_point_2[1],label="l2")
#                 # ax.plot(optimal_point_3[0], optimal_point_3[1],label="l3")
#                 ax.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
#                 ax.set_title("ROC-AUC")
#                 ax.set_xlabel("False Positive Rate")
#                 ax.set_ylabel("True Positive Rate")
#                 ax.legend()
#                 # plt.show()

#                 plt.savefig("roc.jpg")
                ##阈值的调整
                ##观察随阈值而变化的各项指标

                # plot_threshold(list(np.arange(0,1,0.1)),test_pred,y_preob)

                # plt.show()

                    # annot=True，显示各个cell上的数字
            # if(test_acc>0.63):
            #         test_pred ,y_preob= test_y_,test_y
            #         cm = sm.confusion_matrix(y_preob, test_pred)
            #         print("---------------混淆矩阵\n", cm)
            #         print(y_preob)
            #         # print(\n)
            #         print(test_pred)
            #         print(len(probs),len(y_preob))
            #         # print(probs)
            #         # print(torch.nn.functional.softmax(torch.Tensor(probs),dim=-1))
            #         # cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            #         # import seaborn as sns
            #         # f,ax=plt.subplots()
            #         # # cm = [[490,28],[36,446]]
            #         # sns.heatmap(cm,annot=True,vmax=1,square = True,cmap = "Blues")#画热力图
            #         # ax.set_title('confusion matrix') #标题
            #         # ax.set_xlabel('predict') #x轴
            #         # ax.set_ylabel('true') #y轴
            #         # # plt.show()
            #         # plt.savefig("heatcls.jpg")
            #         print(len(test_pred))
            #         cp = sm.classification_report(y_preob, test_pred)
            #         print("---------------分类报告\n", cp)
            #         acc = np.sum(test_pred == y_preob)/len(test_pred)
            #         # print(acc)



            else:
                
                early_stop += 1
                if early_stop >= self.args.early_stop:
                    print("[*] Early stopping at epoch {}.".format(epoch_idx))
                    return
        print('[*] Ending Training')

    def evaluate_on_test(self):
        print('[#] Begin to evaluate on test set')
        self.model.load_state_dict(self.best_ckpt["embedding"])
        _, test_acc, test_f1_mac, test_y, test_y_,probs = self._evaluate('Test')
        
        
        self.update_best_ckpt(None, None, test_acc, test_f1_mac, test_y, test_y_)
        
        self.save()


def main():
    trainer = Trainer(args)
    if args.ckpt_name:
        print("已存在，直接加载")
        trainer.load()
    trainer.train()
    trainer.evaluate_on_test()


if __name__ == '__main__':
    main()

