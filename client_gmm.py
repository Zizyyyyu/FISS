import torch
import torch.nn.functional as F

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

def extract_old_class_features(feature_map,labels,old_logits,old_classes,pseudo_threshold):
    if int(old_classes)<=1:
        return {}
    if old_logits.dim()!=4:
        raise ValueError('old_logits must have shape [B, old_classes, Ho, Wo]')
    if old_logits.shape[0]!=feature_map.shape[0]:
        raise ValueError("old_logits and feature_map have different batch sizes")
    if old_logits.shape[1]<int(old_classes):
        raise ValueError("old_logits do not contain all old classes")
    if not 0.0<=float(pseudo_threshold)<=1.0:
        raise ValueError("pseudo_threshold must be between 0 and 1")
    labels_small=resize_labels_to_feature_map(labels=labels,feature_map=feature_map)
    feature_grid=feature_map_to_grid(feature_map)
    old_logits_small=F.interpolate(
        old_logits.detach().float(),
        size=feature_map.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    old_probabilities=torch.softmax(old_logits_small,dim=1)
    pseudo_confidences, pseudo_labels=(old_probabilities.max(dim=1))
    eligible_old_region=(labels_small<int(old_classes))
    old_class_features = {}
    for class_id in range(1,int(old_classes)):
        class_mask=(eligible_old_region&(pseudo_labels==class_id)&(pseudo_confidences>=float(pseudo_threshold)))
        if not torch.any(class_mask):
            continue
        selected_features=feature_grid[class_mask]
        selected_features=selected_features.detach().float()
        old_class_features[class_id]=selected_features
    return old_class_features

def extract_batch_class_features(current_model,old_model,images,labels,old_classes,nb_current_classes,pseudo_threshold):
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
            old_classes_features=extract_old_class_features(feature_map=feature_map,labels=labels,old_logits=old_logits,old_classes=old_classes,pseudo_threshold=pseudo_threshold)
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
