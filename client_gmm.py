import torch
import torch.nn.functional as F
from fiss_pseudo import get_fiss_entropy_pseudo_labels
from utils.gmm import GaussianMixture

def resize_labels_to_feature_map(labels,feature_map):
    """
    Resize segmentation labels to the spatial size of feature map
    labels: LongTensor [B,H,W]
    feature_map: FloatTensor [B,C,Hf,Wf]
    return LongTensor [B,Hf,Wf]
    [B, H, W]
    ↓ unsqueeze(1)
    [B, 1, H, W]
    ↓ nearest resize
    [B, 1, Hf, Wf]
    ↓ squeeze(1)
    [B, Hf, Wf]
    """
    if labels.dim()!=3:
        raise ValueError(f'labels must have [B,H,W], but we have got {tuple(labels.shape)}')
    if feature_map.dim()!=4:
        raise ValueError(f'feature map must have [B,C,Hf,Wf], but we have got {tuple(feature_map.shape)}')
    if labels.shape[0]!=feature_map.shape[0]:
        raise ValueError(f'labels and features have different batch size')
    feature_size=feature_map.shape[-2:]
    labels_small=F.interpolate(
        labels.unsqueeze(1).float(),size=feature_size,mode='nearest'
    )#B,H,W -> B,1,H,W
    return labels_small.squeeze(1).long()#B,H,W

def feature_map_to_grid(feature_map):
    """
    Change feature layout for boolean-mask indexing
    return: [B,Hf,Wf,C]
    """
    if feature_map.dim()!=4:
        raise ValueError(f'feature map must have B,C,H,W, but we have got {tuple(feature_map.shape)}')
    return feature_map.permute(0,2,3,1).contiguous()

def extract_new_class_features(feature_map,labels,old_classes,nb_current_classes):
    labels_small=resize_labels_to_feature_map(labels=labels,feature_map=feature_map)
    feature_grid=feature_map_to_grid(feature_map)
    first_new_classes=max(1,int(old_classes))
    new_classes_features={}
    for class_id in range(first_new_classes,int(nb_current_classes)):
        class_mask=labels_small==class_id
        if not torch.any(class_mask):
            continue
        selected_features=feature_grid[class_mask]
        selected_features=selected_features.detach().float()
        new_classes_features[class_id]=selected_features
    return new_classes_features

def extract_old_class_features(feature_map,labels,old_logits,old_classes,pseudo_thresholds,max_entropy):
    if int(old_classes)<=1:
        return {}
    if old_logits.dim()!=4:
        raise ValueError('old_logits must have shape [B, old_classes, Ho, Wo]')
    if old_logits.shape[0]!=feature_map.shape[0]:
        raise ValueError("old_logits and feature_map have different batch sizes")
    if old_logits.shape[1]<int(old_classes):
        raise ValueError("old_logits do not contain all old classes")
    labels_small=resize_labels_to_feature_map(labels=labels,feature_map=feature_map)
    feature_grid=feature_map_to_grid(feature_map)
    old_logits_small=F.interpolate(
        old_logits.detach().float(),
        size=feature_map.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    pseudo_labels,valid_pseudo=get_fiss_entropy_pseudo_labels(
        old_logits=old_logits_small,
        thresholds=pseudo_thresholds,
        max_entropy=max_entropy
    )
    eligible_old_region=(labels_small<int(old_classes))
    old_class_features = {}
    for class_id in range(1,int(old_classes)):
        class_mask=(eligible_old_region&(pseudo_labels==class_id)&valid_pseudo)
        if not torch.any(class_mask):
            continue
        selected_features=feature_grid[class_mask]
        selected_features=selected_features.detach().float()
        old_class_features[class_id]=selected_features
    return old_class_features

def extract_batch_class_features(current_model,old_model,images,labels,old_classes,nb_current_classes,pseudo_thresholds,max_entropy):
    if images.dim()!=4:
        raise ValueError(f'images must have 4 dimensions B,3,H,W')
    if labels.dim()!=3:
        raise ValueError(f'labels must have 3 dimensions B,H,W')
    if images.shape[0]!=labels.shape[0]:
        raise ValueError('images and labels have different batch sizes')
    with torch.no_grad():
        _,current_intermediate=current_model(images,ret_intermediate=True)
        if "pre_logits" not in current_intermediate:
            raise KeyError(f"current model didn't return pre_logits")
        feature_map=current_intermediate["pre_logits"]
        new_classes_features=extract_new_class_features(feature_map=feature_map,labels=labels,old_classes=old_classes,nb_current_classes=nb_current_classes)
        old_classes_features={}
        if int(old_classes)>1:
            if old_model is None:
                raise ValueError(f'old model is required')
            _,old_intermediate=old_model(images,ret_intermediate=True)
            if "sem_logits_small" not in old_intermediate:
                raise KeyError(f'old model didnt return sem_logits_small')
            old_logits=old_intermediate['sem_logits_small']
            old_classes_features=extract_old_class_features(feature_map=feature_map,labels=labels,old_logits=old_logits,old_classes=old_classes,pseudo_thresholds=pseudo_thresholds,max_entropy=max_entropy)
        batch_cls_data={}
        for class_id,class_features in old_classes_features.items():
            batch_cls_data[class_id]={
                'features':class_features,
                'label_source':'pseudo'
            }
        for class_id,class_features in new_classes_features.items():
            batch_cls_data[class_id]={
                'features':class_features,
                'label_source':'ground_truth'
            }
        return batch_cls_data

def update_class_feature_accumulator(class_accumulator,batch_cls_data,max_features):
    max_features=int(max_features)
    if max_features<=0:
        raise ValueError(f'max_features must be greater than 0')
    for class_id,class_payload in batch_cls_data.items():
        class_id=int(class_id)
        #tensor: [N,C]
        batch_features=class_payload['features']
        label_source=str(class_payload["label_source"])
        if not torch.is_tensor(batch_features):
            raise TypeError('features must be a tensor')
        if batch_features.dim()!=2:
            raise ValueError('batch_features must have shape [N,C]')
        if label_source not in {'pseudo','ground_truth'}:
            raise ValueError(f'Invalid label source, expect to be pseudo or ground_truth, but we have got {label_source}')
        batch_features_count=int(batch_features.shape[0])
        if batch_features_count==0:
            continue
        batch_features=batch_features.detach().float().cpu()
        batch_sample_keys=torch.rand(batch_features_count)
        if class_id not in class_accumulator:
            class_accumulator[class_id]={
                'feature_count':batch_features_count,
                'features':batch_features,
                'sample_keys':batch_sample_keys,
                'label_source':label_source
            }
        else:
            class_entry=class_accumulator[class_id]
            if class_entry['label_source']!=label_source:
                raise ValueError(f'class {class_id} has inconsistent label sources')
            if class_entry['features'].shape[1]!=batch_features.shape[1]:
                raise ValueError(f'class {class_id} has inconsistent feature dimensions')
            class_entry['feature_count']+=batch_features_count
            class_entry['features']=torch.cat(
                [class_entry['features'],batch_features],dim=0
            )
            class_entry['sample_keys']=torch.cat(
                [class_entry['sample_keys'],batch_sample_keys],dim=0
            )
        class_entry=class_accumulator[class_id]
        if class_entry['features'].shape[0]>max_features:
            keep_indices=torch.topk(class_entry['sample_keys'],k=max_features,largest=False).indices
            class_entry["features"]=class_entry["features"][keep_indices]
            class_entry["sample_keys"]=class_entry["sample_keys"][keep_indices]
    return class_accumulator


def collect_client_class_features(current_model,old_model,data_loader,device,old_classes,nb_current_classes,pseudo_thresholds,max_entropy,max_features):
    if current_model is None:
        raise ValueError('current_model cannot be None')
    if data_loader is None:
        raise ValueError('data_loader cannot be None')
    if int(old_classes)>1 and old_model is None:
        raise ValueError('old_model cannot be None when old classes exist')
    class_accumulator={}
    current_model_was_training=current_model.training
    old_model_was_training=None
    current_model.eval()
    if old_model is not None:
        old_model_was_training=old_model.training
        old_model.eval()
    try:
        for images,labels in data_loader:
            images=images.to(device,dtype=torch.float32,non_blocking=True)
            labels=labels.to(device,dtype=torch.long,non_blocking=True)

            batch_cls_data=extract_batch_class_features(
                current_model=current_model,
                old_model=old_model,
                images=images,
                labels=labels,
                old_classes=old_classes,
                nb_current_classes=nb_current_classes,
                pseudo_thresholds=pseudo_thresholds,
                max_entropy=max_entropy
            )

            class_accumulator=update_class_feature_accumulator(
                class_accumulator=class_accumulator,
                batch_cls_data=batch_cls_data,
                max_features=max_features
            )
        return class_accumulator
    finally:
        current_model.train(current_model_was_training)
        if old_model is not None:
            old_model.train(old_model_was_training)

def fit_class_gmms(class_accumulator,n_components,min_features,em_iters,device):
    n_components=int(n_components)
    min_features=int(min_features)
    em_iters=int(em_iters)
    if n_components<=0:
        raise ValueError('n_components must be greater than 0')
    if min_features<=0:
        raise ValueError('min_features must be greater than 0')
    if em_iters<=0:
        raise ValueError('em_iters must be greater than 0')
    classes_payload={}
    required_features=max(min_features,n_components)
    for class_id in sorted(class_accumulator):
        class_id=int(class_id)
        class_entry=class_accumulator[class_id]
        features=class_entry['features']
        if not torch.is_tensor(features):
            raise TypeError('features must be a tensor')
        if features.dim()!=2:
            raise ValueError('features must have shape [N,C]')
        feature_count=int(class_entry['feature_count'])
        fit_count=int(features.shape[0])
        label_source=str(class_entry['label_source'])
        class_payload:dict={
            'feature_count':feature_count,
            'fit_count':fit_count,
            'label_source':label_source,
            'gmm':None
        }
        if fit_count<required_features:
            classes_payload[class_id]=class_payload
            continue
        n_features=int(features.shape[1])
        fit_features=features.to(device,dtype=torch.float32)
        gmm=GaussianMixture(
            n_components=n_components,
            n_features=n_features,
            covariance_type='diag'
        ).to(device)
        gmm.fit(fit_features,n_iter=em_iters)
        class_payload['gmm']={
            'n_components':n_components,
            'n_features':n_features,
            'covariance_type':'diag',
            'state_dict':{
                'mu':gmm.mu.detach().cpu().clone(),
                'var':gmm.var.detach().cpu().clone(),
                'pi':gmm.pi.detach().cpu().clone()
            }
        }
        classes_payload[class_id]=class_payload
        del fit_features
        del gmm
    return classes_payload

def build_client_gmm_upload(client_id,current_step,class_accumulator,n_components,min_features,em_iters,device):
    client_id=int(client_id)
    current_step=int(current_step)
    if client_id<0:
        raise ValueError('client_id cannot be negative')
    if current_step<0:
        raise ValueError('current_step cannot be negative')
    classes_payload=fit_class_gmms(
        class_accumulator=class_accumulator,
        n_components=n_components,
        min_features=min_features,
        em_iters=em_iters,
        device=device
    )
    client_upload={
        'client_id':client_id,
        'step':current_step,
        'classes':classes_payload
    }
    return client_upload
