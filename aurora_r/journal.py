"""Durable control-action journal. Write-ahead before Scale API calls.

An ambiguous API failure is kept unresolved (not retried automatically).
On restart, reconcile the spec with the recorded requested target. Settlement
requires actual Ready/serving state AND post-convergence SLO observations.
"""
import json,os,sqlite3,time,uuid

UNRESOLVED=('created','applied','ready','inconclusive')
# Statuses that actually block a new decision. ``inconclusive`` is a terminal
# state that needs an operator (see ``cli.py resolve``); it used to block too,
# so ONE ambiguous Scale API call wedged the controller forever -- every later
# cycle reported "unresolved action" and never proposed anything again.
BLOCKING=('created','applied','ready')
class ActionJournal:
    def __init__(self,path):
        os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
        self.path=path
        self.db=sqlite3.connect(path,timeout=10,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS actions(
            id TEXT PRIMARY KEY, service TEXT NOT NULL, created REAL NOT NULL,
            expected INTEGER NOT NULL, target INTEGER NOT NULL,
            status TEXT NOT NULL, ready_at REAL, settled_at REAL,
            before_json TEXT NOT NULL, after_json TEXT, detail TEXT)''')
        self.db.execute('CREATE INDEX IF NOT EXISTS action_status ON actions(status,service)')
    def create(self,service,expected,target,before,now=None):
        now=now or time.time();aid=str(uuid.uuid4())
        self.db.execute('INSERT INTO actions(id,service,created,expected,target,status,before_json) VALUES(?,?,?,?,?,?,?)',
                        (aid,service,now,int(expected),int(target),'created',json.dumps(before,sort_keys=True)))
        return aid
    def update(self,aid,status,now=None,after=None,detail=None):
        if status not in ('created','applied','ready','settled','inconclusive','cancelled'):
            raise ValueError('invalid action state')
        now=now or time.time()
        if status=='ready':
            self.db.execute('UPDATE actions SET status=?,ready_at=COALESCE(ready_at,?),detail=? WHERE id=?',
                            (status,now,detail,aid))
        elif status=='settled':
            self.db.execute('UPDATE actions SET status=?,settled_at=?,after_json=?,detail=? WHERE id=?',
                            (status,now,json.dumps(after),detail,aid))
        elif status=='applied':
            # A readiness regression invalidates the prior observation start.
            self.db.execute('UPDATE actions SET status=?,ready_at=NULL,detail=? WHERE id=?',
                            (status,detail,aid))
        else:
            self.db.execute('UPDATE actions SET status=?,detail=? WHERE id=?',(status,detail,aid))
    def get(self,aid):
        r=self.db.execute('SELECT id,service,created,expected,target,status,ready_at,settled_at,before_json,after_json,detail FROM actions WHERE id=?',(aid,)).fetchone()
        return self._dict(r)
    @staticmethod
    def _dict(r):
        if not r:return None
        keys=['id','service','created','expected','target','status','ready_at','settled_at','before','after','detail']
        d=dict(zip(keys,r))
        for k in ('before','after'):
            d[k]=json.loads(d[k]) if d[k] else None
        return d
    def unresolved(self):
        placeholders=','.join('?' for _ in UNRESOLVED)
        rows=self.db.execute(f'''SELECT id,service,created,expected,target,status,ready_at,settled_at,before_json,after_json,detail
            FROM actions WHERE status IN ({placeholders}) ORDER BY created''',UNRESOLVED)
        return [self._dict(r) for r in rows]
    def blocking(self):
        placeholders=','.join('?' for _ in BLOCKING)
        rows=self.db.execute(f'''SELECT id,service,created,expected,target,status,ready_at,settled_at,before_json,after_json,detail
            FROM actions WHERE status IN ({placeholders}) ORDER BY created''',BLOCKING)
        return [self._dict(r) for r in rows]
    def reconcile_created(self,states):
        for a in self.unresolved():
            if a['status']!='created':continue
            rep=states.get(a['service'])
            if rep is None:continue
            current=rep.spec
            if current==a['target']:
                self.update(a['id'],'applied',detail='recovered: Scale target observed after restart')
            elif current==a['expected']:
                # No confirmed write. Do not silently issue a duplicate.
                self.update(a['id'],'inconclusive',detail='ambiguous interrupted write; operator must resolve')
            else:
                self.update(a['id'],'inconclusive',detail=f'external modification {current}, expected {a["expected"]} or {a["target"]}')
    def operator_resolve(self,action_id,reason):
        """Explicit human reconciliation, never automatic for ambiguous writes."""
        if not reason or len(reason.strip())<8:raise ValueError('reason needs detail')
        a=self.get(action_id)
        if a is None or a['status']!='inconclusive':
            raise ValueError('only inconclusive actions can be explicitly cancelled')
        self.update(action_id,'cancelled',detail='operator: '+reason)

    def last_settled(self,n=100):
        rows=self.db.execute('SELECT id,service,created,expected,target,status,ready_at,settled_at,before_json,after_json,detail FROM actions WHERE status="settled" ORDER BY settled_at DESC LIMIT ?',(n,))
        return [self._dict(r) for r in rows]
    def close(self):self.db.close()