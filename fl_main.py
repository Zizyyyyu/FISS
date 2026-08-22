import copy
from datetime import timedelta
from FC import FC_model
from FC_cogamid import FC_model_CoGaMiD
from federated_gmm import update_global_gmm_pool
import torch
import random
import os.path as osp
import os
import numpy as np
from myNetwork import make_model
from myNetwork_cogamid import make_model_cogamid
from myNetwork_rcil import make_model_rcil

from Fed_utils import * 
from option import args_parser, modify_command_options

from apex import amp
from apex.parallel import DistributedDataParallel
from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import tasks

from dataset import (AdeSegmentationIncremental,
                     VOCSegmentationIncremental, transform)
from metrics import StreamSegMetrics

from rcil_utils import *


def get_testset(opts, step):
    """ Dataset And Augmentation
    """
    test_transform = transform.Compose(
        [
            transform.Resize(size=opts.crop_size),
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    labels, labels_old, _ = tasks.get_task_labels(opts.dataset, opts.task, step)
    labels_cum = labels_old + labels 

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    else:
        raise NotImplementedError

    # if opts.overlap:
    #     path_base += "-ov" 
    

    # if not os.path.exists(path_base):
    #     os.makedirs(path_base, exist_ok=True)


    image_set = 'train' if opts.val_on_trainset else 'val'  
    test_dst = dataset(
        root=opts.data_root,
        train=opts.val_on_trainset, 
        transform=test_transform,
        labels=list(labels_cum),  
        disable_background=opts.disable_background,
        test_on_val=opts.test_on_val,
        step=step,
        ignore_test_bg=opts.ignore_test_bg
    )

    return test_dst, len(labels_cum)


def select_coverage_gmm_clients(client_reports,gmm_clients,old_classes,nb_current_classes):
    gmm_clients=int(gmm_clients)
    if gmm_clients<=0:
        raise ValueError('gmm_clients must be greater than 0')
    reports={}
    for report in client_reports:
        if report is None:
            continue
        client_id=int(report['client_id'])
        reports[client_id]={int(class_id):int(count) for class_id,count in report.get('class_counts',{}).items() if int(count)>0}
    if len(reports)==0:
        raise ValueError('client_reports do not contain any valid client')
    selection_count=min(len(reports),gmm_clients)
    selected=[]
    available=set(reports)

    def add_clients_for_coverage(target_classes):
        uncovered={class_id for class_id in target_classes if any(report.get(class_id,0)>0 for report in reports.values())}
        for client_id in selected:
            uncovered={class_id for class_id in uncovered if reports[client_id].get(class_id,0)<=0}
        while len(uncovered)>0 and len(selected)<selection_count and len(available)>0:
            best_client=max(
                available,
                key=lambda client_id:(
                    sum(reports[client_id].get(class_id,0)>0 for class_id in uncovered),
                    sum(reports[client_id].get(class_id,0) for class_id in uncovered),
                    sum(reports[client_id].values()),
                    -client_id
                )
            )
            covered={class_id for class_id in uncovered if reports[best_client].get(class_id,0)>0}
            if len(covered)==0:
                break
            selected.append(best_client)
            available.remove(best_client)
            uncovered-=covered

    new_class_start=max(1,int(old_classes))
    add_clients_for_coverage(range(new_class_start,int(nb_current_classes)))
    add_clients_for_coverage(range(1,int(old_classes)))
    while len(selected)<selection_count and len(available)>0:
        best_client=max(
            available,
            key=lambda client_id:(sum(reports[client_id].values()),-client_id)
        )
        selected.append(best_client)
        available.remove(best_client)
    return selected


def collect_global_gmm_pool(args,models,model_g,old_global_model,global_gmm_pool,current_step,num_clients,rank,device):
    gmm_pool_path=f"{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_gmm_pool_step_{current_step}.pth"
    distributed.barrier()
    if rank==0:
        print(f'building client GMM uploads for step {current_step}')
        python_rng_state=random.getstate()
        numpy_rng_state=np.random.get_state()
        torch_rng_state=torch.get_rng_state()
        cuda_rng_state=torch.cuda.get_rng_state(device)
        client_uploads=[]
        try:
            selection_count=min(int(num_clients),int(args.gmm_clients))
            if selection_count==int(num_clients):
                gmm_client_ids=list(range(num_clients))
            else:
                print(f'collect lightweight class coverage from {num_clients} clients')
                client_reports=[]
                for client_id in range(num_clients):
                    client_report=models[client_id].build_gmm_coverage_report(
                        args=args,
                        old_global_model=old_global_model,
                        current_step=current_step
                    )
                    client_reports.append(client_report)
                    print(f'client {client_id} GMM coverage report finished')
                per_task_classes=tasks.get_per_task_classes(args.dataset,args.task,current_step)
                old_classes=sum(per_task_classes[:-1])
                nb_current_classes=sum(per_task_classes)
                gmm_client_ids=select_coverage_gmm_clients(
                    client_reports=client_reports,
                    gmm_clients=args.gmm_clients,
                    old_classes=old_classes,
                    nb_current_classes=nb_current_classes
                )
                report_by_client={int(report['client_id']):report.get('class_counts',{}) for report in client_reports}
                available_classes={int(class_id) for report in client_reports for class_id,count in report.get('class_counts',{}).items() if int(count)>0}
                selected_classes={int(class_id) for client_id in gmm_client_ids for class_id,count in report_by_client[client_id].items() if int(count)>0}
                available_new={class_id for class_id in available_classes if class_id>=max(1,old_classes)}
                available_old={class_id for class_id in available_classes if 0<class_id<old_classes}
                print(f'GMM client coverage: new {len(selected_classes&available_new)}/{len(available_new)}, old {len(selected_classes&available_old)}/{len(available_old)}')
                missing_new=sorted(available_new-selected_classes)
                missing_old=sorted(available_old-selected_classes)
                if len(missing_new)>0:
                    print(f'warning: selected GMM clients miss new classes {missing_new}')
                if len(missing_old)>0:
                    print(f'warning: selected GMM clients miss old classes {missing_old}')
                del client_reports
            print(f'select {len(gmm_client_ids)} clients only for GMM construction')
            print(gmm_client_ids)
            for client_id in gmm_client_ids:
                client_upload=models[client_id].build_gmm_upload(
                    args=args,
                    current_global_model=model_g,
                    old_global_model=old_global_model,
                    current_step=current_step
                )
                client_uploads.append(client_upload)
                print(f'client {client_id} GMM upload finished')
            global_gmm_pool,gmm_winners=update_global_gmm_pool(
                global_gmm_pool=global_gmm_pool,
                client_uploads=client_uploads,
                current_step=current_step
            )
            torch.save(global_gmm_pool,gmm_pool_path)
            print(f'global GMM pool updated with {len(gmm_winners)} class winners')
            print(f'global GMM pool saved to {gmm_pool_path}')
            del client_uploads
            del gmm_winners
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state,device)
    distributed.barrier()
    for client_id in range(num_clients):
        models[client_id].release_step_dataset(current_step)
    if rank!=0:
        global_gmm_pool=torch.load(gmm_pool_path,map_location='cpu')
    for client_model in models:
        client_model.set_gmm_pool(global_gmm_pool)
    return global_gmm_pool



def main(args):

    if float(args.distributed_timeout_hours)<=0:
        raise ValueError('distributed_timeout_hours must be greater than 0')
    distributed.init_process_group(backend='nccl',init_method='env://',timeout=timedelta(hours=float(args.distributed_timeout_hours)))
    device_id=int(os.environ.get('LOCAL_RANK',args.local_rank))
    args.local_rank=device_id
    device=torch.device('cuda',device_id)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    setup_seed(args.seed) 
    use_cogamid=args.incremental_method=='CoGaMiD'
    final_task_step=tasks.get_task_steps(args.dataset,args.task)-1
    resume_step=int(args.resume_step)
    rebuild_gmm_step=int(args.rebuild_gmm_step)
    if resume_step<0:
        raise ValueError('resume_step cannot be negative')
    if resume_step>0 and not use_cogamid:
        raise ValueError('resume_step is currently supported only for CoGaMiD')
    if rebuild_gmm_step>=0 and not use_cogamid:
        raise ValueError('rebuild_gmm_step is currently supported only for CoGaMiD')
    if rebuild_gmm_step>=0 and resume_step>0:
        raise ValueError('rebuild_gmm_step and resume_step cannot be used together')
    previous_step=resume_step-1
    initial_model_step=rebuild_gmm_step if rebuild_gmm_step>=0 else previous_step if resume_step>0 else 0

    args.inital_nb_classes = tasks.get_per_task_classes(args.dataset,args.task,step=0)[0] 

    if use_cogamid:
        model_g=make_model_cogamid(args,classes=tasks.get_per_task_classes(args.dataset,args.task,step=initial_model_step))
    elif args.name != 'RCIL':
        model_g = make_model(args, classes=tasks.get_per_task_classes(args.dataset, args.task, step=0)) 
    else:
        model_g = make_model_rcil(args, classes=tasks.get_per_task_classes(args.dataset, args.task, step=0)) 
    
    if args.fix_bn: 
        model_g.fix_bn()


    num_clients=args.num_clients
    models = []
    for client_index in range(40): 
        if use_cogamid:
            model_temp=FC_model_CoGaMiD(client_index,args.batch_size,args.num_workers,args.loss_de,args.pod,world_size,rank,device,args.entropy_threshold)
        else:
            model_temp=FC_model(client_index,args.batch_size,args.num_workers,args.loss_de,args.pod,world_size,rank,device,args.entropy_threshold)
        models.append(model_temp)

    old_global_model=None
    global_gmm_pool={}

    old_step=-1
    start_global_epoch=0
    if rebuild_gmm_step>=0:
        current_step=rebuild_gmm_step
        model_rebuild_path=f'{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{current_step}.pth'
        if not os.path.exists(model_rebuild_path):
            raise FileNotFoundError(f'GMM rebuild model checkpoint does not exist: {model_rebuild_path}')
        model_g.load_state_dict(torch.load(model_rebuild_path,map_location='cpu'),strict=True)
        num_clients=args.num_clients+args.add_clients*current_step
        if num_clients>len(models):
            raise ValueError('rebuild_gmm_step requires more clients than the configured client model capacity')
        if current_step>0:
            previous_model_path=f'{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{current_step-1}.pth'
            previous_gmm_path=f'{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_gmm_pool_step_{current_step-1}.pth'
            if not os.path.exists(previous_model_path):
                raise FileNotFoundError(f'previous model checkpoint does not exist: {previous_model_path}')
            if not os.path.exists(previous_gmm_path):
                raise FileNotFoundError(f'previous GMM pool does not exist: {previous_gmm_path}')
            old_global_model=make_model_cogamid(args,classes=tasks.get_per_task_classes(args.dataset,args.task,step=current_step-1))
            old_global_model.load_state_dict(torch.load(previous_model_path,map_location='cpu'),strict=True)
            old_global_model.eval()
            for param in old_global_model.parameters():
                param.requires_grad=False
            global_gmm_pool=torch.load(previous_gmm_path,map_location='cpu')
        if rank==0:
            print(f'rebuild only the GMM pool for completed step {current_step}')
            print(f'loaded model checkpoint from {model_rebuild_path}')
        global_gmm_pool=collect_global_gmm_pool(
            args=args,
            models=models,
            model_g=model_g,
            old_global_model=old_global_model,
            global_gmm_pool=global_gmm_pool,
            current_step=current_step,
            num_clients=num_clients,
            rank=rank,
            device=device
        )
        distributed.barrier()
        return
    if resume_step>0:
        model_resume_path=f'{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{previous_step}.pth'
        gmm_resume_path=f'{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_gmm_pool_step_{previous_step}.pth'
        if not os.path.exists(model_resume_path):
            raise FileNotFoundError(f'resume model checkpoint does not exist: {model_resume_path}')
        if not os.path.exists(gmm_resume_path):
            raise FileNotFoundError(f'resume GMM pool does not exist: {gmm_resume_path}')
        model_g.load_state_dict(torch.load(model_resume_path,map_location='cpu'),strict=True)
        global_gmm_pool=torch.load(gmm_resume_path,map_location='cpu')
        for client_model in models:
            client_model.set_gmm_pool(global_gmm_pool)
        num_clients=args.num_clients+args.add_clients*max(0,resume_step-1)
        if num_clients+args.add_clients>len(models):
            raise ValueError('resume_step requires more clients than the configured client model capacity')
        old_step=previous_step
        start_global_epoch=resume_step*args.steps_global
        if rank==0:
            print(f'resume CoGaMiD from step {resume_step}, global round {start_global_epoch}')
            print(f'loaded model checkpoint from {model_resume_path}')
            print(f'loaded GMM pool from {gmm_resume_path}')

    for ep_g in range(start_global_epoch,args.epochs_global):

        current_step = ep_g // args.steps_global
        is_task_final_round=((ep_g+1)%args.steps_global==0)
        is_final_task_step=current_step==final_task_step
        should_build_gmm=is_task_final_round and (not is_final_task_step or args.fit_final_gmm)

        if current_step != old_step: 
            test_dst, n_classes = get_testset(args, current_step)
            test_loader = data.DataLoader(
                test_dst,
                batch_size=args.batch_size if args.crop_val else 1, 
                sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                num_workers=args.num_workers
            )
            val_metrics = StreamSegMetrics(n_classes)


        if current_step != old_step and old_step != -1:
            if use_cogamid:
                old_global_model=copy.deepcopy(model_g)
                old_global_model.eval()
                for param in old_global_model.parameters():
                    param.requires_grad=False
            args.base_weights = False
    
            for i in range(num_clients):
                models[i].last_entropy = -1
            num_clients = num_clients + args.add_clients

            if use_cogamid:
                model_g1=make_model_cogamid(args,classes=tasks.get_per_task_classes(args.dataset,args.task,current_step))
                model_g1.load_state_dict(model_g.state_dict(),strict=False)
                model_g=model_g1
            elif args.name != 'RCIL':
                model_g1 = make_model(args, classes=tasks.get_per_task_classes(args.dataset, args.task, current_step)) 
                model_g1.load_state_dict(model_g.state_dict(), strict=False)  
                if args.init_balanced:
                    model_g1.init_new_classifier(device)
                model_g = model_g1
            else:
                model_g1 = make_model_rcil(args, classes=tasks.get_per_task_classes(args.dataset, args.task, current_step)) 
                # add the bias to the left branch STEP > 0

                for name, mm in model_g1.named_modules():
                    if hasattr(mm, 'convs'):
                        mm.convs.conv2.bias = nn.Parameter(torch.zeros(mm.convs.conv2.weight.shape[0]).to(mm.convs.conv2.weight.device))
                    if hasattr(mm, 'map_convs'):
                        for kk in range(4):
                            mm.map_convs[kk].bias = nn.Parameter(torch.zeros(mm.map_convs[kk].weight.shape[0]).to(mm.map_convs[kk].weight.device))

                model_g1.load_state_dict(model_g.state_dict(), strict=False)  

                if args.init_balanced:
                    model_g1.init_new_classifier(device)
                model_g = model_g1

                ###### merge parameters to the left branch #####
                model_g = convert_model(model_g, None)


        if rank==0:
            print('federated global round: {}, step: {}'.format(ep_g, current_step))

        w_local = []


        clients_index = random.sample(range(num_clients), args.local_clients) 

        if rank==0:
            print('select part of clients to conduct local training') 
            print(clients_index)

        for c in clients_index:
            local_model = local_train(args, models, c, model_g, current_step, ep_g)
            w_local.append(local_model)


        
        if rank==0:
            print('federated aggregation...')

        if args.base_weights == False:
            w_g_new = FedAvg(w_local)  
            model_g.load_state_dict(w_g_new) 

            val_score = model_global_eval(args, model_g, test_loader, current_step, val_metrics, device, rank)
        else:
            if ((ep_g+1)% args.steps_global)==0:
                base_ckpt_path = f"{args.checkpoint}/{args.dataset}_{args.task}_base_step_0.pth"
                w_g_new = torch.load(base_ckpt_path)
                model_g.load_state_dict(w_g_new) 
                val_score = model_global_eval(args, model_g, test_loader, current_step, val_metrics, device, rank)
        if rank==0 and is_task_final_round:
            with open(f"{args.results_path}/{args.date}_{args.dataset}_{args.task}_{args.name}.csv","a+") as f:
                classes_iou=','.join(
                    [str(val_score['Class IoU'].get(c,'x')) for c in range(args.num_classes)]
                )
                f.write(f"{current_step},{classes_iou},{val_score['Mean IoU']}\n")
            step_checkpoint_path=f"{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{current_step}.pth"
            torch.save(model_g.state_dict(),step_checkpoint_path)
            print(f'global model saved to {step_checkpoint_path} before GMM construction')
            if current_step==0 and args.name!="RCIL" and args.base_weights==False:
                torch.save(model_g.state_dict(),f"{args.checkpoint}/{args.dataset}_{args.task}_base_step_{current_step}.pth")
        if use_cogamid and is_task_final_round and is_final_task_step and not args.fit_final_gmm and rank==0:
            print(f'skip final step {current_step} GMM construction because no later task will use it')
        if use_cogamid and should_build_gmm:
            global_gmm_pool=collect_global_gmm_pool(
                args=args,
                models=models,
                model_g=model_g,
                old_global_model=old_global_model,
                global_gmm_pool=global_gmm_pool,
                current_step=current_step,
                num_clients=num_clients,
                rank=rank,
                device=device
            )
        if use_cogamid and is_task_final_round:
            distributed.barrier()
        old_step = current_step



if __name__ == '__main__':
    
    args = args_parser() 
    args = modify_command_options(args)

    args.results_path = f"results/seed_{args.seed}"
    args.checkpoint = f"{args.checkpoint}/seed_{args.seed}"

    if args.overlap:
        args.results_path += "-ov"
        args.checkpoint += "-ov"

    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.checkpoint, exist_ok=True) 
    
    main(args)
