"""V3 isolated artifacts and a preservation baseline covering all earlier work."""
import sys,json,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hybrid_v2.embeddinggemma2 import digest,identity
from hybrid_v2.embeddinggemma2_index import write_json
OUT=ROOT/'outputs/driveevidence_v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def preservation(verify=False):
    before=OUT/'hashes/before.json'
    if verify:
        files=read(before)['files'];changed=[]
        for i,(n,h) in enumerate(files.items(),1):
            if not (ROOT/n).is_file() or digest(ROOT/n)!=h:changed.append(n)
            if i%1000==0:print('V3 protected hashes checked',i,flush=True)
        write_json(OUT/'hashes/after.json',{'files':len(files),'changed':changed,'status':'FAIL' if changed else 'PASS','time_unix':time.time()})
        if changed:raise ValueError('Protected asset changed '+str(changed))
        return
    if before.exists():return
    # Earlier baseline was fully reverified after V2. Preserve those exact bytes.
    prior=read(ROOT/'outputs/driveevidence_v2/hashes/after.json')
    if prior['status']!='PASS':raise ValueError('Previous preservation audit failed')
    files=read(ROOT/'outputs/driveevidence_v2/hashes/before.json')['files'].copy()
    additions=list((ROOT/'outputs/driveevidence_v2').rglob('*'))+list(ROOT.glob('CODEX_*REPORT.md'))
    additions+=list((ROOT/'hybrid_v2').glob('driveevidence_v2*.py'))+list((ROOT/'scripts').glob('*driveevidence_v2*.py'))
    additions+=[ROOT/'docs/driveevidence-v2.md',ROOT/'tests/laptop/test_driveevidence_v2.py']
    for p in additions:
        if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=digest(p)
    write_json(before,{'files':files,'prior_after_sha256':digest(ROOT/'outputs/driveevidence_v2/hashes/after.json'),'time_unix':time.time()})
    manifest=read(ROOT/'outputs/driveevidence_v2/dev/corpus/manifest.json')
    write_json(OUT/'dev/corpus/manifest.json',manifest)
    write_json(OUT/'hashes/benchmark_gates.json',{'pilot_queries':digest(ROOT/'outputs/metadata_v4_1_pilot50/pilot_queries.json'),
      'pilot_selection':digest(ROOT/'outputs/metadata_v4_1_pilot50/pilot_selection.json'),
      'old_full_queries':digest(ROOT/'outputs/roadscapes_hf_200_e2e/query_benchmark/queries.json'),
      'new_heldout_exists':(ROOT/'outputs/driveevidence_v2/heldout/benchmark_frozen.json').exists(),
      'v2_new_heldout_run':False,'training_performed':False,'fine_tuning_performed':False})
    print('V3 preservation baseline',len(files),flush=True)
if __name__=='__main__':preservation('--verify' in sys.argv)
