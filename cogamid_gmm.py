import torch
from utils.gmm import GaussianMixture


def restore_gmm(gmm_payload,device):
    if gmm_payload is None:
        raise ValueError('gmm_payload cannot be None')
    state_dict=gmm_payload['state_dict']
    gmm=GaussianMixture(
        n_components=int(gmm_payload['n_components']),
        n_features=int(gmm_payload['n_features']),
        covariance_type=str(gmm_payload['covariance_type']),
        mu_init=state_dict['mu'].detach().float().clone(),
        var_init=state_dict['var'].detach().float().clone()
    ).to(device)
    gmm.pi.data.copy_(state_dict['pi'].detach().float().to(device))
    gmm.params_fitted=True
    return gmm


def restore_old_gmm_pool(global_gmm_pool,old_classes,device):
    old_gmms={}
    feature_counts={}
    for class_id,class_payload in global_gmm_pool.items():
        class_id=int(class_id)
        if class_id<=0 or class_id>=int(old_classes):
            continue
        if class_payload.get('gmm') is None:
            continue
        old_gmms[class_id]=restore_gmm(class_payload['gmm'],device)
        feature_counts[class_id]=int(class_payload['feature_count'])
    return old_gmms,feature_counts


def get_gmm_prototypes(old_gmms):
    prototypes=[]
    for class_id in sorted(old_gmms):
        gmm=old_gmms[class_id]
        prototype=(gmm.mu.squeeze(0)*gmm.pi.squeeze(0)).sum(dim=0)
        prototypes.append(prototype)
    if len(prototypes)==0:
        return None
    return torch.stack(prototypes,dim=0)


def allocate_uniform_replay_counts(class_ids,total_budget):
    class_ids=sorted(int(class_id) for class_id in class_ids)
    total_budget=int(total_budget)
    if total_budget<0:
        raise ValueError('total_budget cannot be negative')
    if len(class_ids)==0:
        return {}
    base_count=total_budget//len(class_ids)
    remainder=total_budget%len(class_ids)
    return {
        class_id:base_count+int(index<remainder)
        for index,class_id in enumerate(class_ids)
    }


def calculate_client_extra_background_ratio(predicted_background,predicted_old_total,new_feature_total,max_ratio):
    predicted_background=float(predicted_background)
    predicted_old_total=float(predicted_old_total)
    new_feature_total=float(new_feature_total)
    max_ratio=float(max_ratio)
    if min(predicted_background,predicted_old_total,new_feature_total,max_ratio)<0:
        raise ValueError('feature counts and max_ratio cannot be negative')
    extra_background_target=max(0.0,predicted_old_total-new_feature_total)
    return min(max_ratio,extra_background_target/max(1.0,predicted_background))


def sample_old_gmm_features(old_gmms,sample_counts,noise_scale,return_labels=False):
    sampled_features=[]
    sampled_labels=[]
    for class_id in sorted(old_gmms):
        sample_count=int(sample_counts.get(class_id,0))
        if sample_count<=0:
            continue
        class_features=old_gmms[class_id].sample(sample_count)[0]
        if float(noise_scale)>0:
            class_features=class_features+torch.randn_like(class_features)*float(noise_scale)
        sampled_features.append(class_features)
        sampled_labels.append(torch.full((sample_count,),int(class_id),dtype=torch.long,device=class_features.device))
    if len(sampled_features)==0:
        return (None,None) if return_labels else None
    sampled_features=torch.cat(sampled_features,dim=0)
    sampled_features=sampled_features.transpose(0,1).unsqueeze(0).unsqueeze(3)
    if not return_labels:
        return sampled_features
    return sampled_features,torch.cat(sampled_labels,dim=0)
