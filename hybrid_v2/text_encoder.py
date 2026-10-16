
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BGEEncoder:
    def __init__(self,model_name="BAAI/bge-base-en-v1.5",device="cuda",max_length=128,
                 cache_dir=None,revision=None,local_files_only=False):
        from .local_runtime import resolve_model, model_identity
        self.model_name=model_name
        self.device=torch.device(device if device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length=max_length
        root=resolve_model(model_name,cache_dir,revision,local_files_only)
        self.identity=model_identity(model_name,root)
        self.tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
        self.model=AutoModel.from_pretrained(root,local_files_only=True).to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self,texts,batch_size=128):
        texts=list(texts); outs=[]
        for s in range(0,len(texts),batch_size):
            x=self.tok(texts[s:s+batch_size],padding=True,truncation=True,
                       max_length=self.max_length,return_tensors="pt")
            x={k:v.to(self.device) for k,v in x.items()}
            y=self.model(**x).last_hidden_state[:,0].float()
            y=F.normalize(y,dim=-1)
            outs.append(y.cpu().numpy())
        if not outs:
            return np.zeros((0,int(self.model.config.hidden_size)),dtype=np.float32)
        return np.concatenate(outs,axis=0).astype(np.float32)
