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


def sample_old_gmm_features(old_gmms,sample_counts,noise_scale):
    sampled_features=[]
    for class_id in sorted(old_gmms):
        sample_count=int(sample_counts.get(class_id,0))
        if sample_count<=0:
            continue
        class_features=old_gmms[class_id].sample(sample_count)[0]
        if float(noise_scale)>0:
            class_features=class_features+torch.randn_like(class_features)*float(noise_scale)
        sampled_features.append(class_features)
    if len(sampled_features)==0:
        return None
    sampled_features=torch.cat(sampled_features,dim=0)
    return sampled_features.transpose(0,1).unsqueeze(0).unsqueeze(3)
