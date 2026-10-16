"""Cheap pre-exact move ranking with feature-distance / ensemble OOD fallback.

Training uses ONLY officially PASS teacher runs supplied by the training
manifest. Exact-preview fields are removed. PPO is an explicitly one-step
candidate bandit control, not a long-horizon manufacturing policy.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(28,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class Ranker:
    def __init__(self,models,mean,std,limit,kind='ranker'):
        self.models=models; self.mean=np.asarray(mean,dtype=np.float32); self.std=np.asarray(std,dtype=np.float32); self.limit=limit; self.kind=kind
        self.calls=0; self.ood_calls=0
    def score(self,features):
        self.calls+=1
        x=(np.asarray(features,dtype=np.float32)[:,:28]-self.mean)/self.std
        distance=float(np.max(np.abs(x)))
        with torch.no_grad(): predictions=np.asarray([m(torch.tensor(x,dtype=torch.float32)).numpy() for m in self.models])
        mean=predictions.mean(axis=0); spread=float(predictions.std(axis=0).max())
        ood=distance>self.limit or (spread>max(.25,float(np.std(mean))*3))
        self.ood_calls+=int(ood)
        return mean.tolist(),ood
    @classmethod
    def load(cls,path):
        torch.set_num_threads(1)
        data=torch.load(path,map_location='cpu',weights_only=False)
        models=[]
        for weights in data['weights']:
            m=Network(); m.load_state_dict(weights); m.eval(); models.append(m)
        return cls(models,np.asarray(data['mean']),np.asarray(data['std']),data['ood_limit'],data['kind'])

def train(teacher_paths,output):
    from .search import FEATURE_NAMES
    torch.set_num_threads(1); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    records=[]; sources=[]
    for path in teacher_paths:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('status')!='PASS' or not data.get('official',{}).get('passed'): raise ValueError('Non-PASS teacher rejected')
        sources.append(str(path))
        for record in data['search']['records']:
            records.append({**record,'source':str(path)})
    if not records: raise ValueError('No valid teacher transitions')
    # Entire contexts stay together; never split competing candidates between
    # train and validation. No public or unseen synthetic source is loaded.
    contexts=sorted(set((r['source'],r['context']) for r in records))
    rng=np.random.default_rng(711); rng.shuffle(contexts)
    held=set(contexts[::5]); train_ids=[i for i,r in enumerate(records) if (r['source'],r['context']) not in held]
    val_ids=[i for i,r in enumerate(records) if (r['source'],r['context']) in held]
    raw=np.asarray([r['features'][:28] for r in records],dtype=np.float32)
    mean=raw[train_ids].mean(axis=0); std=np.maximum(raw[train_ids].std(axis=0),.05)
    x=torch.tensor((raw-mean)/std); y=torch.tensor([r['relative_gain']*100 for r in records],dtype=torch.float32)
    weights=[]; losses=[]
    for seed in (711,719):
        torch.manual_seed(seed); model=Network(); opt=torch.optim.Adam(model.parameters(),lr=.002)
        best=None; best_loss=float('inf')
        for step in range(600):
            ids=torch.tensor(rng.choice(train_ids,min(128,len(train_ids)),replace=True))
            loss=((model(x[ids])-y[ids])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step%20==0:
                with torch.no_grad(): validation=float(((model(x[val_ids])-y[val_ids])**2).mean())
                if validation<best_loss:
                    best_loss=validation; best={k:v.detach().clone() for k,v in model.state_dict().items()}
        weights.append(best); losses.append(best_loss)
    metadata={'mean':mean.tolist(),'std':std.tolist(),'ood_limit':float(max(6.,np.max(np.abs(x[train_ids].numpy()))*1.25)),
              'weights':weights,'kind':'ranker','training_sources':sources,'records':len(records),'contexts':len(contexts),
              'validation_mse':losses,'feature_names':FEATURE_NAMES,'heldout_protocol':'train-instance context holdout; separate synthetic family test untouched'}
    torch.save(metadata,output/'ranker.pt')
    # PPO control starts from supervised scorer; train on fixed one-step
    # candidate outcomes. No exact outcomes are input features at inference.
    model=Network(); model.load_state_dict(weights[0]); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    groups={c:[i for i,r in enumerate(records) if (r['source'],r['context'])==c] for c in contexts if c not in held}
    groups=[g for g in groups.values() if len(g)>1]; clipped=[]
    for step in range(200):
        group=groups[step%len(groups)] if groups else train_ids[:2]
        xx=x[group]; rewards=y[group]
        with torch.no_grad():
            old=torch.distributions.Categorical(logits=model(xx)); action=old.sample(); log_old=old.log_prob(action)
            advantage=rewards[action]-torch.sum(old.probs*rewards)
        for epoch in range(3):
            dist=torch.distributions.Categorical(logits=model(xx)); ratio=torch.exp(dist.log_prob(action)-log_old)
            loss=-torch.minimum(ratio*advantage,torch.clamp(ratio,.8,1.2)*advantage)-.005*dist.entropy()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    torch.save({**metadata,'weights':[model.state_dict()],'kind':'ppo_one_step_bandit','ppo_updates':600},output/'ppo.pt')
    report={k:v for k,v in metadata.items() if k!='weights'}
    report['ppo_scope']='one-step offline candidate bandit, not full-episode PPO'; report['ppo_updates']=600
    (output/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
