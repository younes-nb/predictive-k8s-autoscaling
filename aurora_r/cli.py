"""AURORA-R command line: capture, train, run, inspect and reconcile actions."""
import argparse,json,time
from pathlib import Path

def _cfg(path):
    from .config import Config
    return Config.load(path)

def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='train':
        from .train.train import main as train_main
        return train_main(argv[1:])
    ap=argparse.ArgumentParser(prog='aurora-r')
    sub=ap.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='run the horizontal controller (shadow by default)')
    run.add_argument('--config',required=True)
    run.add_argument('--once',action='store_true')
    cap=sub.add_parser('capture',help='record full sparse graph snapshots without requiring a checkpoint')
    cap.add_argument('--config',required=True)
    cap.add_argument('--output',required=True)
    cap.add_argument('--cycles',type=int,default=1)
    cap.add_argument('--run-id',default='')
    init=sub.add_parser('init-shadow',help='create an UNTRAINED artifact solely for plumbing tests')
    init.add_argument('--output',required=True)
    init.add_argument('--dimension',type=int,default=48)
    list_a=sub.add_parser('actions',help='inspect persistent action journal')
    list_a.add_argument('--journal',required=True)
    resolve=sub.add_parser('resolve',help='operator-cleared inconclusive Scale API action')
    resolve.add_argument('--journal',required=True)
    resolve.add_argument('--action-id',required=True)
    resolve.add_argument('--reason',required=True)
    labels=sub.add_parser('attach-labels',help='attach externally reviewed root/remedy/failure labels to captured snapshots')
    labels.add_argument('--snapshots',required=True)
    labels.add_argument('--labels',required=True)
    labels.add_argument('--output',required=True)
    export=sub.add_parser('export-interventions',help='convert settled action journal to intervention JSONL')
    export.add_argument('--journal',required=True)
    export.add_argument('--review',default='')
    export.add_argument('--output',required=True)
    bench=sub.add_parser('benchmark',help='CPU scaling benchmark over synthetic sparse graphs')
    bench.add_argument('--output',default='')
    args=ap.parse_args(argv)
    if args.command=='init-shadow':
        import torch
        from .model import SparseTemporalGraph,ActionResponseModel,checkpoint_dict
        d=SparseTemporalGraph(dim=args.dimension); r=ActionResponseModel(dim=args.dimension)
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(checkpoint_dict(d,r,{'untrained':True,'outcome_target':'delta_rates',
            'intervention_validated':False,'heldout_deployment_validation':False,
            'probability_calibrated':False,'usable_interventions':0,
            'warning':'RANDOM WEIGHTS: NOT A RESEARCH OR LIVE SCALING MODEL'}),path)
        print('Untrained SHADOW artifact written:',path)
        return
    if args.command in ('actions','resolve'):
        from .journal import ActionJournal
        j=ActionJournal(args.journal)
        if args.command=='actions':
            for a in j.unresolved()+j.last_settled():print(json.dumps(a,sort_keys=True))
        else:
            j.operator_resolve(args.action_id,args.reason)
            print('Recorded operator resolution; verify live Deployment state before another action.')
        j.close();return
    if args.command=='attach-labels':
        from .datasets import annotate_snapshots
        print('Annotated',annotate_snapshots(args.snapshots,args.labels,args.output),'snapshots')
        return
    if args.command=='export-interventions':
        from .datasets import export_interventions
        n,approved=export_interventions(args.journal,args.review,args.output)
        print(f'Exported {n} settled actions; {approved} passed explicit experiment evidence checks')
        return
    if args.command=='benchmark':
        from .benchmark import run
        data=run()
        if args.output:Path(args.output).write_text(json.dumps(data,indent=2))
        print(json.dumps(data,indent=2));return
    cfg=_cfg(args.config)
    if args.command=='run':
        from .controller import AuroraController
        ctl=AuroraController(cfg)
        ctl.run(once=args.once);return
    if args.command=='capture':
        from .kube import KubernetesReader
        from .metrics import TelemetryCollector
        kube=KubernetesReader(cfg);collector=TelemetryCollector(cfg)
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        run_id=args.run_id or str(int(time.time()))
        with path.open('a',encoding='utf-8') as fh:
            for i in range(args.cycles):
                start=time.monotonic();states=kube.read_states()
                snap,stats=collector.collect(run_id,states)
                fh.write(json.dumps(snap.to_dict(),allow_nan=False)+'\n')
                fh.flush()
                print(json.dumps({'capture':i+1,'services':len(snap.services),
                                  'edges':snap.edge_index.shape[1],'stats':stats}))
                if i<args.cycles-1:
                    time.sleep(max(0,cfg.decision_interval_s-(time.monotonic()-start)))
        return
if __name__=='__main__':main()