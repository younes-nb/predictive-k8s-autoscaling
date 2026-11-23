"""Cautious deployment-local outcome calibration without full retraining.

This is an exponentially discounted residual corrector. It is not causal proof
and never trains the shared encoder online. It only learns from completed,
workload-matched actions; estimates shrink toward zero for low sample count.
"""
from pathlib import Path
import json,math,os,tempfile

class OnlineAdapter:
    def __init__(self,path,prior=10,forget=.985):
        self.path=Path(path);self.prior=prior;self.forget=forget
        try:self.state=json.loads(self.path.read_text())
        except (OSError,ValueError):self.state={}
    def correction(self,service,direction):
        s=self.state.get(f'{service}:{direction}',{})
        n=float(s.get('n',0))
        return [float(v)*n/(n+self.prior) for v in s.get('residual',[0.0,0.0])]
    def update(self,service,direction,predicted,measured,matched,settled):
        if not (matched and settled):return False
        vals=[float(measured[i])-float(predicted[i]) for i in (0,1)]
        if not all(math.isfinite(v) for v in vals):return False
        key=f'{service}:{direction}'
        s=self.state.setdefault(key,{'n':0.0,'residual':[0.0,0.0]})
        n=s['n']*self.forget
        alpha=1.0/(n+1.0)
        # Bound individual contributions to resist sharp single-action shifts.
        s['residual']=[(1-alpha)*float(old)+alpha*max(-.25,min(.25,v))
                       for old,v in zip(s['residual'],vals)]
        s['n']=n+1
        self.persist();return True
    def persist(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',dir=self.path.parent,delete=False) as f:
            json.dump(self.state,f,sort_keys=True)
            f.flush();os.fsync(f.fileno());tmp=f.name
        os.replace(tmp,self.path)