import math
import torch
from torch import distributed


def calculate_fiss_entropy(probabilities):
    if probabilities.dim()!=4:
        raise ValueError('probabilities must have shape [B,C,H,W]')
    if probabilities.shape[1]<=1:
        raise ValueError('probabilities must contain at least two classes')
    factor=1/math.log(probabilities.shape[1]+1e-8)
    return -factor*torch.mean(probabilities*torch.log(probabilities+1e-8),dim=1)


def get_fiss_entropy_pseudo_labels(old_logits,thresholds,max_entropy):
    if old_logits.dim()!=4:
        raise ValueError('old_logits must have shape [B,C,H,W]')
    if thresholds.dim()!=1:
        raise ValueError('thresholds must have shape [C]')
    if thresholds.shape[0]<old_logits.shape[1]:
        raise ValueError('thresholds do not contain all old classes')
    if float(max_entropy)<=0:
        raise ValueError('max_entropy must be greater than 0')
    old_probabilities=torch.softmax(old_logits.float(),dim=1)
    pseudo_labels=old_probabilities.argmax(dim=1)
    normalized_entropy=calculate_fiss_entropy(old_probabilities)/max_entropy
    valid_pseudo=normalized_entropy<thresholds[pseudo_labels]
    return pseudo_labels,valid_pseudo


def find_fiss_entropy_thresholds(old_model,data_loader,old_classes,nb_current_classes,target_portion,device,base_threshold=0.001,nb_bins=100,sync_distributed=True):
    old_classes=int(old_classes)
    nb_current_classes=int(nb_current_classes)
    target_portion=float(target_portion)
    base_threshold=float(base_threshold)
    nb_bins=int(nb_bins)
    if old_model is None:
        raise ValueError('old_model cannot be None')
    if data_loader is None:
        raise ValueError('data_loader cannot be None')
    if old_classes<=1:
        raise ValueError('old_classes must be greater than 1')
    if nb_current_classes<old_classes:
        raise ValueError('nb_current_classes cannot be smaller than old_classes')
    if not 0.0<=target_portion<=1.0:
        raise ValueError('target_portion must be between 0 and 1')
    if base_threshold<0.0:
        raise ValueError('base_threshold cannot be negative')
    if nb_bins<=0:
        raise ValueError('nb_bins must be greater than 0')
    max_entropy=torch.log(torch.tensor(float(nb_current_classes),device=device))
    histograms=torch.zeros((old_classes,nb_bins),dtype=torch.long,device=device)
    old_model_was_training=old_model.training
    old_model.eval()
    try:
        with torch.no_grad():
            for images,labels in data_loader:
                images=images.to(device,dtype=torch.float32,non_blocking=True)
                labels=labels.to(device,dtype=torch.long,non_blocking=True)
                old_logits,_=old_model(images,ret_intermediate=True)
                mask_background=labels==0
                if not torch.any(mask_background):
                    continue
                old_probabilities=torch.softmax(old_logits.float(),dim=1)
                pseudo_labels=old_probabilities.argmax(dim=1)[mask_background]
                normalized_entropy=(calculate_fiss_entropy(old_probabilities)/max_entropy)[mask_background]
                bin_indices=torch.clamp((normalized_entropy*nb_bins).long(),min=0,max=nb_bins-1)
                flat_indices=pseudo_labels*nb_bins+bin_indices
                histograms+=torch.bincount(flat_indices,minlength=old_classes*nb_bins).view(old_classes,nb_bins)
        if sync_distributed and distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(histograms)
        thresholds=torch.full((old_classes,),base_threshold,dtype=torch.float32,device=device)
        for class_id in range(old_classes):
            class_histogram=histograms[class_id]
            total=class_histogram.sum()
            if total<=0:
                continue
            target=total.float()*target_portion
            cumulative=torch.cumsum(class_histogram,dim=0)
            bin_index=int(torch.searchsorted(cumulative,target,right=False).clamp(max=nb_bins-1).item())
            previous=cumulative[bin_index-1] if bin_index>0 else torch.tensor(0,device=device)
            bin_count=class_histogram[bin_index].clamp(min=1)
            fraction=(target-previous.float())/bin_count.float()
            threshold=(bin_index+fraction.clamp(min=0.0,max=1.0))/nb_bins
            thresholds[class_id]=torch.maximum(threshold,thresholds[class_id])
        return thresholds,max_entropy
    finally:
        old_model.train(old_model_was_training)
