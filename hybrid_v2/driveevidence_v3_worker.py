"""The unchanged frozen EG2 protocol, with exclusively V3-owned logs."""
import os,subprocess
from .driveevidence_worker import DriveEvidenceWorker,main
from .embeddinggemma2 import ROOT,cached_model_path,worker_python_path
class DriveEvidenceV3Worker(DriveEvidenceWorker):
    def load(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:return self
            cached_model_path()
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
            log=ROOT/'outputs/driveevidence_v3/logs/worker.stderr.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a',encoding='utf-8') as stderr:
                self.process=subprocess.Popen([str(worker_python_path()),'-B','-m','hybrid_v2.driveevidence_v3_worker'],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
            try:self.metrics=self.request('load',config=self.config)
            except Exception:self.close();raise
            return self
if __name__=='__main__':main()
