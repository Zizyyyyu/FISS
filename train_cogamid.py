import torch
import torch.nn.functional as F
from apex import amp
from torch import distributed

from cogamid_gmm import get_gmm_prototypes,restore_old_gmm_pool,sample_old_gmm_features
from cogamid_losses import CoGaMiDContrastLoss,CoGaMiDPKDLoss,CoGaMiDWeightedBCELoss,calculate_cogamid_certainty


def unwrap_cogamid_model(model):
    return model.module if hasattr(model,'module') else model


def classify_cogamid_features(model,feature_map):
    network=unwrap_cogamid_model(model)
    if hasattr(network,'classify_features'):
        classifier_dtype=next(network.cls.parameters()).dtype
        with amp.disable_casts():
            return network.classify_features(feature_map.to(dtype=classifier_dtype))
    outputs=[]
    for index,classifier in enumerate(network.cls):
        if index==0 and network.multi_modal_background:
            outputs.append(network.fusion(classifier(feature_map)))
        elif network.use_cosine:
            weight=F.normalize(classifier.weight,dim=1,p=2)
            outputs.append(F.conv2d(feature_map,weight))
        else:
            outputs.append(classifier(feature_map))
    return torch.cat(outputs,dim=1)


class Trainer_CoGaMiD:
    def __init__(self,model,model_old,device,rank,opts,classes,step,gmm_pool):
        self.model=model
        self.model_old=model_old
        self.device=device
        self.rank=rank
        self.opts=opts
        self.classes=classes
        self.step=int(step)
        self.old_classes=sum(classes[:-1])
        self.nb_current_classes=sum(classes)
        self.new_class_start=1 if self.step==0 else self.old_classes
        self.new_class_count=self.nb_current_classes-self.new_class_start
        self.mbce_loss=CoGaMiDWeightedBCELoss(
            new_class_start=self.new_class_start,
            new_class_count=self.new_class_count,
            pos_weight=opts.cogamid_pos_weight
        ).to(device)
        self.pkd_loss=CoGaMiDPKDLoss().to(device)
        self.contrast_loss=CoGaMiDContrastLoss(
            new_class_start=self.new_class_start,
            new_class_count=self.new_class_count
        ).to(device)
        self.old_gmms={}
        self.old_feature_counts={}
        self.old_prototypes=None
        if self.step>0:
            self.old_gmms,self.old_feature_counts=restore_old_gmm_pool(gmm_pool,self.old_classes,device)
            self.old_prototypes=get_gmm_prototypes(self.old_gmms)
            if self.rank==0 and len(self.old_gmms)<self.old_classes-1:
                print(f'CoGaMiD GMM pool contains {len(self.old_gmms)}/{self.old_classes-1} old classes')

    def before(self,cur_epoch,train_loader):
        return None

    def state_dict(self):
        return {}

    def _make_fake_features(self,num_batches):
        return sample_old_gmm_features(
            old_gmms=self.old_gmms,
            feature_counts=self.old_feature_counts,
            num_batches=num_batches,
            max_per_class=self.opts.cogamid_replay_max_per_class,
            noise_scale=self.opts.cogamid_feature_noise
        )

    def _make_extra_background_features(self,features,labels,old_logits):
        labels_small=F.interpolate(labels.unsqueeze(1).float(),size=features.shape[-2:],mode='nearest').squeeze(1).long()
        old_logits_small=F.interpolate(old_logits.detach(),size=features.shape[-2:],mode='bilinear',align_corners=False)
        old_prediction=old_logits_small.argmax(dim=1)
        region_background=(labels_small==0)&(old_prediction==0)
        if not torch.any(region_background):
            return None
        feature_grid=features.permute(0,2,3,1)
        selected_features=feature_grid[region_background]
        return selected_features.transpose(0,1).unsqueeze(0).unsqueeze(3)

    def train(self,cur_epoch,optim,train_loader,scheduler=None,print_int=10):
        if len(train_loader)==0:
            raise ValueError('CoGaMiD train_loader is empty')
        if self.rank==0:
            print(f'Epoch {cur_epoch+1}, lr = {optim.param_groups[0]["lr"]}')
        train_loader.sampler.set_epoch(cur_epoch)
        self.model.train()
        unwrap_cogamid_model(self.model).in_eval=False
        if self.model_old is not None:
            self.model_old.eval()
            unwrap_cogamid_model(self.model_old).in_eval=False
        class_loss_sum=0.0
        regularization_loss_sum=0.0
        interval_loss=0.0
        for cur_step,(images,labels) in enumerate(train_loader):
            images=images.to(self.device,dtype=torch.float32,non_blocking=True)
            labels=labels.to(self.device,dtype=torch.long,non_blocking=True)
            optim.zero_grad()
            old_logits=None
            old_features=None
            pseudo_region=None
            if self.step>0:
                if self.model_old is None:
                    raise ValueError('CoGaMiD requires the frozen old model after the base step')
                with torch.no_grad():
                    old_logits,old_intermediate=self.model_old(images,ret_intermediate=True)
                    old_features=old_intermediate['pre_logits']
                    old_probabilities=torch.softmax(old_logits,dim=1)
                    old_confidences,old_predictions=old_probabilities.max(dim=1)
                    pseudo_region=((labels==0)&(old_predictions>0)&(old_confidences>=float(self.opts.gmm_pseudo_threshold))).unsqueeze(1)
            outputs,intermediate=self.model(images,ret_intermediate=True)
            features=intermediate['pre_logits']
            new_logits=outputs[:,self.new_class_start:self.new_class_start+self.new_class_count]
            loss_mbce=self.mbce_loss(new_logits,labels)
            loss_pkd=features.sum()*0.0
            loss_cont=features.sum()*0.0
            loss_uncer=features.sum()*0.0
            if self.step>0:
                current_new_score=new_logits.detach().max(dim=1).values
                old_score=old_logits[:,1:].detach().max(dim=1).values
                negative_mask=((labels==0)&(old_score>current_new_score)).unsqueeze(1)
                pseudo_region=pseudo_region&negative_mask
                fake_features=self._make_fake_features(len(train_loader))
                fake_weight=0.0
                fake_loss=features.sum()*0.0
                if fake_features is not None:
                    fake_logits=classify_cogamid_features(self.model,fake_features)
                    fake_labels=torch.zeros((1,fake_logits.shape[2],fake_logits.shape[3]),dtype=torch.long,device=self.device)
                    fake_loss=self.mbce_loss(fake_logits[:,self.new_class_start:self.new_class_start+self.new_class_count],fake_labels)
                    fake_weight=float(fake_logits.shape[2]*fake_logits.shape[3])/(features.shape[0]*features.shape[2]*features.shape[3])
                extra_background_features=self._make_extra_background_features(features,labels,old_logits)
                extra_background_weight=0.0
                extra_background_loss=features.sum()*0.0
                if extra_background_features is not None:
                    extra_background_logits=classify_cogamid_features(self.model,extra_background_features)
                    extra_background_labels=torch.zeros((1,extra_background_logits.shape[2],extra_background_logits.shape[3]),dtype=torch.long,device=self.device)
                    extra_background_loss=self.mbce_loss(extra_background_logits[:,self.new_class_start:self.new_class_start+self.new_class_count],extra_background_labels)
                    extra_background_weight=float(extra_background_logits.shape[2]*extra_background_logits.shape[3])/(features.shape[0]*features.shape[2]*features.shape[3])
                loss_mbce=(loss_mbce+fake_loss*fake_weight+extra_background_loss*extra_background_weight)/(1.0+fake_weight+extra_background_weight)
                loss_pkd=self.pkd_loss(features,old_features,pseudo_region,self.old_gmms)
                loss_cont=self.contrast_loss(features,labels,self.old_prototypes)
                foreground_logits=outputs[:,1:]
                certainty=calculate_cogamid_certainty(torch.sigmoid(foreground_logits)).squeeze(1)
                foreground=(labels==(foreground_logits.detach().argmax(dim=1)+1))|(torch.sigmoid(foreground_logits.detach()).max(dim=1).values>0.7)
                uncertain_region=(~foreground).float()
                loss_uncer=((1.0-certainty)*uncertain_region).pow(2).mean()
            weighted_mbce=self.opts.cogamid_mbce*loss_mbce
            regularization_loss=self.opts.cogamid_pkd*loss_pkd+self.opts.cogamid_cont*loss_cont+self.opts.cogamid_uncer*loss_uncer
            loss_total=weighted_mbce+regularization_loss
            with amp.scale_loss(loss_total,optim) as scaled_loss:
                scaled_loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            class_loss_sum+=weighted_mbce.item()
            regularization_loss_sum+=regularization_loss.item()
            interval_loss+=loss_total.item()
            if (cur_step+1)%print_int==0 and self.rank==0:
                print(f'Epoch {cur_epoch+1}, Batch {cur_step+1}/{len(train_loader)}, Loss={interval_loss/print_int}')
                print(f'Loss made of: MBCE {weighted_mbce}, PKD {loss_pkd}, Cont {loss_cont}, Uncer {loss_uncer}')
                interval_loss=0.0
        class_loss=torch.tensor(class_loss_sum,device=self.device)
        regularization_loss=torch.tensor(regularization_loss_sum,device=self.device)
        distributed.reduce(class_loss,dst=0)
        distributed.reduce(regularization_loss,dst=0)
        if self.rank==0:
            divisor=distributed.get_world_size()*len(train_loader)
            class_loss=class_loss/divisor
            regularization_loss=regularization_loss/divisor
            print(f'Epoch {cur_epoch+1}, Class Loss={class_loss}, Reg Loss={regularization_loss}')
        return class_loss,regularization_loss
