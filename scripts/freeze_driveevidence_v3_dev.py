"""Freeze source-inspected grades before V3 model evidence or optimization."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hybrid_v2.embeddinggemma2 import digest,identity
from hybrid_v2.embeddinggemma2_index import write_json
P=ROOT/'outputs/driveevidence_v3/dev'
def main():
    target=P/'benchmark_frozen.json'
    if target.exists():
        v=json.loads(target.read_text())
        for name,h in v['hashes'].items():assert digest(P/name)==h,name
        print('Verified frozen development benchmark');return
    q=json.loads((P/'queries/queries.json').read_text())['queries']
    clips=json.loads((P/'corpus/manifest.json').read_text())['clips']
    packets=json.loads((P/'source_review/packets.json').read_text())
    old=json.loads((ROOT/'outputs/driveevidence_v2/dev/queries/queries.json').read_text())['queries']
    for a,b in zip(q[:60],old):
        for k in ('query','accepted_clip_ids','acceptable_global_timestamp_ranges'):assert a[k]==b[k],k
    annotations={}
    for path in sorted((P/'adjudications').glob('batch_*.json')):
        for a in json.loads(path.read_text(encoding='utf-8-sig')):
            assert a['query'] not in annotations
            annotations[a['query']]=a
    assert len(q)==len(packets['records'])==len(annotations)==88
    records=[];eligible=[];conflicts=[]
    for n,(query,packet) in enumerate(zip(q,packets['records']),1):
        a=annotations[n];review=set(packet['review_rows']);r=set(a['relevant']);p=set(a['partial'])
        assert not r&p and (r|p)<=review,(n,(r|p)-review)
        relevant_own=[j for j in r if clips[j]['fold']==query['fold']]
        if relevant_own:eligible.append(query['id'])
        else:conflicts.append({'id':query['id'],'reason':'No source-verified relevant clip in its sequence fold; eligibility fixed before V3 evidence.'})
        records.append({'id':query['id'],'grades':{str(j):('RELEVANT' if j in r else 'PARTIALLY_RELEVANT' if j in p else 'NOT_RELEVANT') for j in sorted(review)},'relevant_rows':sorted(r),'partial_rows':sorted(p),'note':a['note'],'validation_eligible':bool(relevant_own)})
    write_json(P/'queries/graded_relevance.json',{'records':records,'provenance':'Agent directly inspected both five-frame raw-source candidate sheets for every query. Not human certified. No detector, ROI, Laya or V3 prediction used for grading. Unreviewed rows UNKNOWN, never certified negative.','reviewed_pairs':sum(len(x['grades']) for x in records),'partial_is_relevant':False})
    names=['queries/queries.json','queries/graded_relevance.json','corpus/manifest.json','source_review/packets.json']
    names += ['adjudications/'+x.name for x in sorted((P/'adjudications').glob('batch_*.json'))]
    names += ['source_review/'+x for x in packets['hashes']]
    for name,h in packets['hashes'].items():assert digest(P/'source_review'/name)==h
    write_json(target,{'time_unix':time.time(),'hashes':{name:digest(P/name) for name in names},'clips':len(clips),'queries':len(q),'eligible_query_ids':eligible,'predeclared_label_conflicts':conflicts,'folds':3,'protocol':'Sequence-disjoint three-fold query CV. Each query ranks the gallery of its own sequence fold; select parameters using the other two query folds, then validate on the third. Global150 descriptive evaluation separately reported. Promotion only within frozen EG2 Top10; all possible primary top10 swaps are source graded. MRR@10 primary; known-positive full-rank MRR diagnostic. Partial grades do not count as relevant. Eligibility is frozen before any V3 evidence.','optimization_split':'TRAIN','source_target_windows_unchanged':True,'models_frozen':True})
    print(json.dumps({'queries':len(q),'eligible':len(eligible),'conflicts':conflicts,'reviewed_pairs':sum(len(x['grades']) for x in records)}))
if __name__=='__main__':main()
