import copy

"""
client_uploads = [
    {
        "client_id": 0,
        "step": 1,
        "classes": {
            1: {
                "feature_count": 12640,
                "fit_count": 12640,
                "label_source": "pseudo",
                "gmm": {
                    "n_components": 3,
                    "n_features": 256,
                    "covariance_type": "diag",
                    "state_dict": {
                        "mu": CPU_FloatTensor,
                        "var": CPU_FloatTensor,
                        "pi": CPU_FloatTensor
                    }
                }
            },
        }
    },
    {
        "client_id": 1,
        "step": 1,
        "classes": {}
    },
    None]
    """
def select_class_winners(client_uploads,current_step):
    """
    Select one complete client GMM for every class.
    Select rule: To select the largest feature_count
    If more than one client have the same count, we select the one with smaller idx
    """
    winners={}
    for upload in client_uploads:
        if upload is None:
            continue
        upload_step=int(upload['step'])
        if upload_step!=int(current_step):
            continue
        client_id=int(upload['client_id'])
        for class_id,class_payload in upload.get("classes",{}).items():
            class_id=int(class_id)
            feature_count=int(class_payload['feature_count'])

            #A class without a fitted GMM can't participate
            if class_payload.get('gmm') is None:
                continue
            if feature_count<=0:
                continue
            candidate=copy.deepcopy(class_payload)
            candidate['client_id']=client_id
            candidate['step']=upload_step
            old_winner=winners.get(class_id)
            if old_winner is None:
                winners[class_id]=candidate
                continue
            old_count=int(old_winner['feature_count'])
            old_client_id=int(old_winner['client_id'])
            has_more_features=feature_count>old_count
            wins_tie=(feature_count==old_count and client_id<old_client_id)
            if has_more_features or wins_tie:
                winners[class_id]=candidate
    return winners

def update_global_gmm_pool(global_gmm_pool,client_uploads,current_step):
    """
    Refresh the server GMM pool with the winners of the current task.
    Existing classes without a valid current upload are retained and
    marked as stale.
    """
    updated_pool=copy.deepcopy(global_gmm_pool)
    for class_id in updated_pool:
        updated_pool[class_id]['stale']=True
    winners=select_class_winners(
        client_uploads=client_uploads,
        current_step=current_step
    )
    for class_id,winner in winners.items():
        pool_entry=copy.deepcopy(winner)
        pool_entry['stale']=False
        updated_pool[class_id]=pool_entry
    return updated_pool,winners