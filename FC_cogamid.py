import copy
import os
import torch
from apex import amp
from apex.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import tasks
import utils
from FC import FC_model
from client_gmm import build_client_gmm_upload,collect_client_class_features
from myNetwork_cogamid import make_model_cogamid
from train_cogamid import Trainer_CoGaMiD


class FC_model_CoGaMiD(FC_model):
    def __init__(self,client_index,batch_size,num_workers,loss_de,pod,world_size,rank,device,entropy_threshold):
        super().__init__(client_index,batch_size,num_workers,loss_de,pod,world_size,rank,device,entropy_threshold)
        self.gmm_pool={}
        self.old_model_step=-1

    def set_gmm_pool(self,gmm_pool):
        self.gmm_pool=copy.deepcopy(gmm_pool)

    def beforeTrain(self,args,current_step):
        if int(current_step)!=self.learned_step:
            self.trainer_state=None
        super().beforeTrain(args,current_step)

    def train(self,args,model_g,ep_g):
        model=copy.deepcopy(model_g)
        if self.signal:
            self.cur_epoch=0
            if ep_g//args.steps_global==0:
                self.last_learning_rate=args.lr1
            else:
                self.last_learning_rate=args.lr2
        params=[]
        if not args.freeze:
            params.append({
                'params':filter(lambda p:p.requires_grad,model.body.parameters()),
                'weight_decay':args.weight_decay
            })
        params.append({
            'params':filter(lambda p:p.requires_grad,model.head.parameters()),
            'weight_decay':args.weight_decay,
            'lr':self.last_learning_rate*args.cogamid_head_lr_mult
        })
        if self.learned_step>0:
            params.append({
                'params':filter(lambda p:p.requires_grad,model.cls[:-1].parameters()),
                'weight_decay':args.weight_decay,
                'lr':self.last_learning_rate*args.cogamid_old_classifier_lr_mult
            })
            params.append({
                'params':filter(lambda p:p.requires_grad,model.cls[-1:].parameters()),
                'weight_decay':args.weight_decay,
                'lr':self.last_learning_rate*args.cogamid_new_classifier_lr_mult
            })
        else:
            params.append({
                'params':filter(lambda p:p.requires_grad,model.cls.parameters()),
                'weight_decay':args.weight_decay,
                'lr':self.last_learning_rate*args.cogamid_old_classifier_lr_mult
            })
        if model.scalar is not None:
            params.append({'params':[model.scalar],'weight_decay':args.weight_decay})
        optimizer=torch.optim.SGD(params,lr=self.last_learning_rate,momentum=0.9,nesterov=True)
        train_loader=data.DataLoader(
            self.current_trainset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.current_trainset,num_replicas=self.world_size,rank=self.rank),
            num_workers=self.num_workers,
            drop_last=True
        )
        if args.lr_policy=='poly':
            scheduler=utils.PolyLR(
                optimizer,max_iters=(args.epochs_local*(args.steps_global-(ep_g%args.steps_global)))*len(train_loader),power=args.lr_power
            )
        elif args.lr_policy=='step':
            scheduler=torch.optim.lr_scheduler.StepLR(
                optimizer,step_size=args.lr_decay_step,gamma=args.lr_decay_factor
            )
        else:
            raise NotImplementedError
        previous_step=self.learned_step-1
        if self.learned_step>0 and self.old_model_step!=previous_step:
            ckpt_path=f'{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{previous_step}.pth'
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'old CoGaMiD checkpoint does not exist: {ckpt_path}')
            if self.rank==0:
                print('load old CoGaMiD model')
            self.old_model=make_model_cogamid(args,classes=tasks.get_per_task_classes(args.dataset,args.task,previous_step))
            self.old_model.load_state_dict(torch.load(ckpt_path,map_location='cpu'),strict=True)
            self.old_model_step=previous_step
        if self.old_model is not None:
            model_old=copy.deepcopy(self.old_model)
            [model,model_old],optimizer=amp.initialize([model.to(self.device),model_old.to(self.device)],optimizer,opt_level=args.opt_level)
            model_old=DistributedDataParallel(model_old)
            for parameter in model_old.parameters():
                parameter.requires_grad=False
            model_old.eval()
        else:
            model_old=None
            model,optimizer=amp.initialize(model.to(self.device),optimizer,opt_level=args.opt_level)
        model=DistributedDataParallel(model,delay_allreduce=True)
        trainer=Trainer_CoGaMiD(
            model=model,
            model_old=model_old,
            device=self.device,
            rank=self.rank,
            opts=args,
            classes=tasks.get_per_task_classes(args.dataset,args.task,self.learned_step),
            step=self.learned_step,
            gmm_pool=self.gmm_pool,
            trainer_state=self.trainer_state
        )
        for cur_epoch in range(args.epochs_local):
            trainer.before(cur_epoch=self.cur_epoch,train_loader=train_loader)
            self.cur_epoch+=1
            model.train()
            epoch_loss=trainer.train(
                cur_epoch=cur_epoch,
                optim=optimizer,
                train_loader=train_loader,
                scheduler=scheduler
            )
            if self.rank==0:
                print(
                    f'Clinet index {self.client_index}, End of Epoch {cur_epoch+1}/{args.epochs_local}, Average Loss={epoch_loss[0]+epoch_loss[1]},'
                    f' Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}'
                )
        self.trainer_state=trainer.state_dict()
        if args.freeze:
            self.last_learning_rate=optimizer.param_groups[0]['lr']/args.cogamid_head_lr_mult
        else:
            self.last_learning_rate=optimizer.param_groups[0]['lr']
        model=model.to('cpu')
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        if args.use_entropy_detection==False:
            self.signal=False
        local_model=model.module.state_dict()
        del model
        del params
        del optimizer
        del scheduler
        del train_loader
        del trainer
        if model_old is not None:
            model_old=model_old.to('cpu')
            torch.cuda.empty_cache()
            del model_old
        return local_model

    def build_gmm_upload(self,args,current_global_model,old_global_model,current_step):
        if self.rank!=0:
            return None
        current_step=int(current_step)
        if self.learned_step!=current_step:
            self.beforeTrain(args,current_step)
        if self.current_trainset is None:
            raise ValueError(f'client {self.client_index} has no current_trainset')
        per_task_classes=tasks.get_per_task_classes(args.dataset,args.task,current_step)
        old_classes=sum(per_task_classes[:-1])
        nb_current_classes=sum(per_task_classes)
        gmm_batch_size=self.batch_size if args.gmm_batch_size is None else int(args.gmm_batch_size)
        if gmm_batch_size<=0:
            raise ValueError('gmm_batch_size must be greater than 0')
        gmm_loader=data.DataLoader(
            self.current_trainset,
            batch_size=gmm_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        current_model=copy.deepcopy(current_global_model).to(self.device)
        current_model.eval()
        if hasattr(current_model,'in_eval'):
            current_model.in_eval=True
        old_model=None
        if old_classes>1:
            if old_global_model is None:
                raise ValueError('old_global_model cannot be None when old classes exist')
            old_model=copy.deepcopy(old_global_model).to(self.device)
            old_model.eval()
            if hasattr(old_model,'in_eval'):
                old_model.in_eval=True
        try:
            class_accumulator=collect_client_class_features(
                current_model=current_model,
                old_model=old_model,
                data_loader=gmm_loader,
                device=self.device,
                old_classes=old_classes,
                nb_current_classes=nb_current_classes,
                pseudo_threshold=args.gmm_pseudo_threshold,
                max_features=args.gmm_max_features
            )
            client_upload=build_client_gmm_upload(
                client_id=self.client_index,
                current_step=current_step,
                class_accumulator=class_accumulator,
                n_components=args.gmm_components,
                min_features=args.gmm_min_features,
                em_iters=args.gmm_em_iters,
                device=self.device
            )
        finally:
            current_model.to('cpu')
            if old_model is not None:
                old_model.to('cpu')
            torch.cuda.empty_cache()
        return client_upload
