"""Experiment data integrity helpers; no automatic invention of causal labels."""
from pathlib import Path
import json
from .journal import ActionJournal
from .schema import GraphSnapshot,Intervention

DIAG_FIELDS=('root_labels','affected_labels','remedy_labels',
             'scalein_labels','degradation_labels','failure_labels')

def annotate_snapshots(source,labels_file,dest):
    """Attach experimenter-provided labels keyed by (deployment,run,timestamp).

    JSON annotation keys are arbitrary IDs mapping to {deployment_id, run_id,
    timestamp, <label_fields>}. Labels must be recorded using ground truth from
    independently instrumented faults/interventions, not utilization heuristics.
    """
    specs=json.loads(Path(labels_file).read_text())
    keys={}
    for a in specs.values() if isinstance(specs,dict) else specs:
        key=(str(a['deployment_id']),str(a['run_id']),float(a['timestamp']))
        if key in keys:raise ValueError('duplicate annotation for '+str(key))
        keys[key]=a
    count=0
    with open(source,encoding='utf-8') as src,open(dest,'w',encoding='utf-8') as dst:
        for line in src:
            if not line.strip():continue
            rec=json.loads(line)
            snap=GraphSnapshot.from_dict(rec)
            a=keys.get((snap.deployment_id,snap.run_id,snap.timestamp))
            if a:
                for field in DIAG_FIELDS:
                    if field in a:setattr(snap,field,a[field])
                count+=1
            dst.write(json.dumps(snap.validate().to_dict(),allow_nan=False)+'\n')
    return count

def export_interventions(journal_path,review_path,dest):
    """Extract settled actions, preserving OBSERVATIONAL status by default.

    An external experiment-review manifest must explicitly establish whether
    treatment was isolated and whether randomized/matched controls exist; a
    mere pre/post comparison is NOT evidence of causal treatment benefit.
    """
    review=json.loads(Path(review_path).read_text()) if review_path else {}
    journal=ActionJournal(journal_path)
    n=approved=0
    with open(dest,'w',encoding='utf-8') as fh:
        for row in journal.last_settled(n=100000):
            before=row.get('before') or {}; after=row.get('after') or {}
            if 'snapshot' not in before or not all(
                k in after for k in ('post_eligible','post_miss_rate','post_capacity_failure_rate',
                                    'workload_shift')):
                continue
            evidence=review.get(row['id'],{})
            item=Intervention(
                snapshot=GraphSnapshot.from_dict(before['snapshot']),
                service=row['service'],target_replicas=row['target'],
                post_eligible=float(after['post_eligible']),
                post_missed=float(after['post_miss_rate'])*float(after['post_eligible']),
                post_capacity_fail=float(after['post_capacity_failure_rate'])*float(after['post_eligible']),
                workload_shift=float(after['workload_shift']),
                settled=row['status']=='settled',
                isolated=evidence.get('isolated') is True,
                matched_control=evidence.get('matched_control') is True,
                confidence_tag=evidence.get('confidence_tag','observational'),
                control_adjustment_miss=float(evidence.get('control_adjustment_miss',0.)),
                control_adjustment_fail=float(evidence.get('control_adjustment_fail',0.)),
            )
            fh.write(json.dumps(item.to_dict(),allow_nan=False)+'\n')
            n+=1
            approved+=int(item.eligible())
    journal.close()
    return n,approved