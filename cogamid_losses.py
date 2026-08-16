import torch
import torch.nn as nn
import torch.nn.functional as F


class CoGaMiDWeightedBCELoss(nn.Module):
    def __init__(self,new_class_start,new_class_count,pos_weight=None,ignore_index=255):
        super().__init__()
        self.new_class_start=int(new_class_start)
        self.new_class_count=int(new_class_count)
        self.ignore_index=int(ignore_index)
        if pos_weight is None:
            self.register_buffer('pos_weight',None)
        else:
            self.register_buffer('pos_weight',torch.full((self.new_class_count,1,1),float(pos_weight)))

    def forward(self,new_logits,labels):
        if new_logits.dim()!=4:
            raise ValueError('new_logits must have shape [B,C,H,W]')
        if labels.dim()!=3:
            raise ValueError('labels must have shape [B,H,W]')
        if new_logits.shape[1]!=self.new_class_count:
            raise ValueError('new_logits have an unexpected number of classes')
        target=torch.zeros_like(new_logits)
        for class_id in range(self.new_class_start,self.new_class_start+self.new_class_count):
            target[:,class_id-self.new_class_start]=(labels==class_id).float()
        loss=F.binary_cross_entropy_with_logits(new_logits,target,pos_weight=self.pos_weight,reduction='none')
        return loss.mean(dim=(0,2,3)).sum()


class CoGaMiDPKDLoss(nn.Module):
    def forward(self,features,features_old,pseudo_region,old_gmms):
        if features.shape!=features_old.shape:
            raise ValueError('current and old feature maps must have the same shape')
        mask=F.interpolate(pseudo_region.float(),size=features.shape[-2:],mode='bilinear',align_corners=False)
        mask_sum=mask.sum()
        if mask_sum.item()==0:
            return features.sum()*0.0
        feature_loss=(features-features_old).pow(2)
        feature_loss=(feature_loss*mask).sum()/(mask_sum*features.shape[1]+1e-4)
        if len(old_gmms)==0:
            return feature_loss
        current_grid=features.permute(0,2,3,1).reshape(-1,features.shape[1])
        old_grid=features_old.permute(0,2,3,1).reshape(-1,features_old.shape[1])
        current_scores=[]
        old_scores=[]
        for class_id in sorted(old_gmms):
            gmm=old_gmms[class_id]
            current_scores.append(gmm.score_samples(current_grid).reshape(-1,1))
            old_scores.append(gmm.score_samples(old_grid).reshape(-1,1))
        current_scores=torch.nan_to_num(torch.cat(current_scores,dim=1))
        old_scores=torch.nan_to_num(torch.cat(old_scores,dim=1))
        distribution_loss=(F.normalize(current_scores,p=2,dim=1)-F.normalize(old_scores,p=2,dim=1)).pow(2)
        flat_mask=mask.permute(0,2,3,1).reshape(-1,1)
        distribution_loss=(distribution_loss*flat_mask).sum()/(mask_sum*distribution_loss.shape[1]+1e-4)
        return feature_loss+distribution_loss


class CoGaMiDContrastLoss(nn.Module):
    def __init__(self,new_class_start,new_class_count,ignore_index=255):
        super().__init__()
        self.new_class_start=int(new_class_start)
        self.new_class_count=int(new_class_count)
        self.ignore_index=int(ignore_index)

    def forward(self,features,labels,old_prototypes):
        if old_prototypes is None or old_prototypes.numel()==0:
            return features.sum()*0.0
        target=torch.zeros((labels.shape[0],self.new_class_count,labels.shape[1],labels.shape[2]),dtype=features.dtype,device=features.device)
        for class_id in range(self.new_class_start,self.new_class_start+self.new_class_count):
            target[:,class_id-self.new_class_start]=(labels==class_id).float()
        target=F.interpolate(target,size=features.shape[-2:],mode='bilinear',align_corners=False)
        new_features=F.normalize(features,p=2,dim=1).unsqueeze(1)*target.unsqueeze(2)
        new_centers=F.normalize(new_features.sum(dim=(0,3,4)),p=2,dim=1)
        old_prototypes=F.normalize(old_prototypes,p=2,dim=1)
        distances=torch.cdist(old_prototypes,new_centers,p=2)
        return (1.0/distances.min(dim=0).values.clamp_min(1e-6)).mean()


def calculate_cogamid_certainty(segmentation_probabilities):
    if segmentation_probabilities.dim()!=4:
        raise ValueError('segmentation_probabilities must have shape [B,C,H,W]')
    if segmentation_probabilities.shape[1]<2:
        return torch.ones_like(segmentation_probabilities[:,:1])
    top2_scores=torch.topk(segmentation_probabilities,k=2,dim=1).values
    return (top2_scores[:,0].detach()-top2_scores[:,1]).unsqueeze(1)
