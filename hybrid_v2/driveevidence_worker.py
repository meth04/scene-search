"""Isolated frozen EG2 worker with DriveEvidence-owned logs and shared cached frames."""
from __future__ import annotations
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import numpy as np
from .embeddinggemma2 import ROOT, EmbeddingGemma2Encoder, EmbeddingGemma2Worker, worker_python_path, cached_model_path

class DriveEvidenceWorker(EmbeddingGemma2Worker):
    def load(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None: return self
            cached_model_path()
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0',
                 'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
            log=ROOT/'outputs/driveevidence/logs/worker.stderr.log'; log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a',encoding='utf-8') as stderr:
                self.process=subprocess.Popen([str(worker_python_path()),'-B','-m','hybrid_v2.driveevidence_worker'],
                    cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                    text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try: self.metrics=self.request('load',config=self.config)
            except Exception:
                self.close(); raise
            return self
    def encode_frames(self,path,indices=None,box=None):
        result=self.request('frames',path=str(path),indices=indices,box=box)
        return np.asarray(result['embedding'],np.float32),result['metrics']

def main():
    def blocked(*a,**k): raise RuntimeError('DRIVEEVIDENCE_EXTERNAL_NETWORK_PROHIBITED')
    socket.socket.connect=blocked; socket.create_connection=blocked
    encoder=None; cached_path=None; frames=None
    for line in sys.stdin:
        r=json.loads(line); response={'id':r['id']}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if r['operation']=='load':
                    import torch
                    if torch.cuda.device_count()!=1 or 'GTX 1650' not in torch.cuda.get_device_name(0): raise RuntimeError('GTX1650_ONLY')
                    if r['config']['dtype']!='float32': raise ValueError('FLOAT32_REQUIRED')
                    encoder=EmbeddingGemma2Encoder(r['config']).load()
                    encoder.model.requires_grad_(False)
                    result=encoder.metrics
                elif r['operation']=='text':
                    vector,metrics=encoder.encode_text(r['query']); result={'embedding':vector.tolist(),'metrics':metrics}
                elif r['operation']=='frames':
                    if cached_path!=r['path']:
                        frames=np.load(r['path'],allow_pickle=False); cached_path=r['path']
                    selected=frames[r['indices']] if r.get('indices') is not None else frames
                    if r.get('box') is not None:
                        x1,y1,x2,y2=r['box']; selected=selected[:,y1:y2,x1:x2,:]
                    vector,metrics=encoder._encode({'video':np.ascontiguousarray(selected)},
                        processing_kwargs={'video':{'do_sample_frames':False,'max_soft_tokens':140}})
                    result={'embedding':vector.tolist(),'metrics':metrics}
                else: raise ValueError('Unsupported DriveEvidence operation')
            response['result']=result
        except Exception as exc: response['error']={'code':getattr(exc,'code','DRIVEEVIDENCE_WORKER_FAILED'),'detail':repr(exc)}
        print(json.dumps(response),flush=True)
    if encoder: encoder.close()

if __name__=='__main__': main()
