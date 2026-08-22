import pathlib
import sys
import types

import torch


REPOSITORY_ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY_ROOT))

apex=types.ModuleType('apex')
apex.amp=types.SimpleNamespace()
sys.modules.setdefault('apex',apex)

from cogamid_losses import CoGaMiDPKDLoss
from train_cogamid import Trainer_CoGaMiD


class FakeGMM:
    def score_samples(self,features):
        return features.sum(dim=1)


def test_zero_pseudo_mask_keeps_pkd_graph():
    features=torch.randn(2,4,3,3,requires_grad=True)
    features_old=torch.randn_like(features)
    pseudo_region=torch.zeros(2,1,6,6,dtype=torch.bool)
    loss=CoGaMiDPKDLoss()(features,features_old,pseudo_region,{1:FakeGMM()})
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad)==0


def test_zero_background_keeps_classifier_graph():
    features=torch.randn(2,4,3,3,requires_grad=True)
    labels=torch.zeros(2,6,6,dtype=torch.long)
    old_logits=torch.zeros(2,2,6,6)
    old_logits[:,1]=1.0
    selected_features,selected_count=Trainer_CoGaMiD._make_extra_background_features(None,features,labels,old_logits)
    assert selected_count==0
    assert selected_features.shape==(1,4,1,1)
    classifier=torch.nn.Conv2d(4,1,1)
    loss=classifier(selected_features).sum()*float(selected_count)
    loss.backward()
    assert classifier.weight.grad is not None
    assert torch.count_nonzero(classifier.weight.grad)==0
    assert features.grad is not None


def test_present_background_preserves_real_feature_count():
    features=torch.randn(2,4,3,3,requires_grad=True)
    labels=torch.zeros(2,6,6,dtype=torch.long)
    old_logits=torch.zeros(2,2,6,6)
    old_logits[:,0]=1.0
    selected_features,selected_count=Trainer_CoGaMiD._make_extra_background_features(None,features,labels,old_logits)
    assert selected_count==18
    assert selected_features.shape==(1,4,18,1)


if __name__=='__main__':
    test_zero_pseudo_mask_keeps_pkd_graph()
    test_zero_background_keeps_classifier_graph()
    test_present_background_preserves_real_feature_count()
    print('CoGaMiD rank-consistent loss path tests PASSED')
