import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from myNetwork import make_model


class CoGaMiDNetwork(nn.Module):
    def __init__(self,network):
        super().__init__()
        self.network=network

    @property
    def body(self):
        return self.network.body

    @property
    def head(self):
        return self.network.head

    @property
    def cls(self):
        return self.network.cls

    @property
    def scalar(self):
        return self.network.scalar

    @property
    def multi_modal_background(self):
        return False

    @property
    def use_cosine(self):
        return self.network.use_cosine

    @property
    def in_eval(self):
        return self.network.in_eval

    @in_eval.setter
    def in_eval(self,value):
        self.network.in_eval=bool(value)

    def classify_features(self,feature_map):
        outputs=[]
        for classifier in self.network.cls:
            if self.network.use_cosine:
                weight=F.normalize(classifier.weight,dim=1,p=2)
            else:
                weight=classifier.weight
            class_logits=torch.einsum('bchw,oc->bohw',feature_map,weight[:,:,0,0])
            if classifier.bias is not None:
                class_logits=class_logits+classifier.bias.view(1,-1,1,1)
            outputs.append(class_logits)
        foreground_logits=torch.cat(outputs,dim=1)
        background_logits=torch.zeros_like(foreground_logits[:,:1])
        return torch.cat([background_logits,foreground_logits],dim=1)

    def forward(self,x,scales=None,do_flip=False,ret_intermediate=False,only_bg=False):
        if only_bg:
            background_logits=torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]),dtype=x.dtype,device=x.device)
            return background_logits,{}
        foreground_logits,intermediate=self.network(
            x,scales=scales,do_flip=do_flip,ret_intermediate=ret_intermediate,only_bg=False
        )
        background_logits=torch.zeros_like(foreground_logits[:,:1])
        logits=torch.cat([background_logits,foreground_logits],dim=1)
        if ret_intermediate:
            foreground_logits_small=intermediate['sem_logits_small']
            background_logits_small=torch.zeros_like(foreground_logits_small[:,:1])
            intermediate['sem_logits_small']=torch.cat([background_logits_small,foreground_logits_small],dim=1)
        return logits,intermediate

    def fix_bn(self):
        self.network.fix_bn()

    def init_new_classifier(self,device):
        return None

    def align_weight(self,align_type):
        self.network.align_weight(align_type)


def make_model_cogamid(opts,classes):
    if int(opts.nb_background_modes)!=1:
        raise ValueError('CoGaMiD requires nb_background_modes=1')
    foreground_classes=copy.deepcopy(classes)
    if len(foreground_classes)==0 or int(foreground_classes[0])<=1:
        raise ValueError('the base task must contain background and at least one foreground class')
    foreground_classes[0]-=1
    return CoGaMiDNetwork(make_model(opts,classes=foreground_classes))
