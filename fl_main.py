import copy
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



def main(args):

    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id=int(os.environ.get('LOCAL_RANK',args.local_rank))
    args.local_rank=device_id
    device=torch.device('cuda',device_id)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    setup_seed(args.seed) 
    use_cogamid=args.incremental_method=='CoGaMiD'

    args.inital_nb_classes = tasks.get_per_task_classes(args.dataset,args.task,step=0)[0] 

    if use_cogamid:
        model_g=make_model_cogamid(args,classes=tasks.get_per_task_classes(args.dataset,args.task,step=0))
    elif args.name != 'RCIL':
        model_g = make_model(args, classes=tasks.get_per_task_classes(args.dataset, args.task, step=0)) 
    else:
        model_g = make_model_rcil(args, classes=tasks.get_per_task_classes(args.dataset, args.task, step=0)) 
    
    if args.fix_bn: 
        model_g.fix_bn()


    num_clients = args.num_clients 
    models = []
    for client_index in range(40): 
        if use_cogamid:
            model_temp=FC_model_CoGaMiD(client_index,args.batch_size,args.num_workers,args.loss_de,args.pod,world_size,rank,device,args.entropy_threshold)
        else:
            model_temp=FC_model(client_index,args.batch_size,args.num_workers,args.loss_de,args.pod,world_size,rank,device,args.entropy_threshold)
        models.append(model_temp)

    old_global_model=None
    global_gmm_pool={}

    old_step = -1

    for ep_g in range(args.epochs_global): 

        current_step = ep_g // args.steps_global
        is_task_final_round=((ep_g+1)%args.steps_global==0)

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
        if use_cogamid and is_task_final_round:
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
                    for client_id in range(num_clients):
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
            if rank!=0:
                global_gmm_pool=torch.load(gmm_pool_path,map_location='cpu')
            for client_model in models:
                client_model.set_gmm_pool(global_gmm_pool)
        if rank == 0:
            if ((ep_g+1)% args.steps_global)==0:
                with open(f"{args.results_path}/{args.date}_{args.dataset}_{args.task}_{args.name}.csv", "a+") as f:
                    classes_iou = ','.join(
                        [str(val_score['Class IoU'].get(c, 'x')) for c in range(args.num_classes)]
                    )
                    f.write(f"{current_step},{classes_iou},{val_score['Mean IoU']}\n")
                
                torch.save(model_g.state_dict(),  f"{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{current_step}.pth")

                if current_step==0 and args.name != "RCIL" and args.base_weights == False:
                    torch.save(model_g.state_dict(),  f"{args.checkpoint}/{args.dataset}_{args.task}_base_step_{current_step}.pth")

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
