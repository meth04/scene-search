
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def make_projection(state,prefix):
    w=state[f"{prefix}.1.weight"]; out_dim,in_dim=w.shape
    mod=nn.Sequential(nn.LayerNorm(in_dim),nn.Linear(in_dim,out_dim))
    sub={k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+".")}
    mod.load_state_dict(sub,strict=True)
    return mod.eval()

def get_tensor(y):
    if torch.is_tensor(y): return y
    z=getattr(y,"pooler_output",None)
    if z is None: raise TypeError(type(y))
    return z

class SigLIPQueryEncoder:
    def __init__(self,checkpoint,device="cuda",max_length=64,dtype="fp16",
                 cache_dir=None,revision=None,local_files_only=False,model_path=None,text_only=False):
        from .local_runtime import load_checkpoint, resolve_model, model_identity
        ck=load_checkpoint(checkpoint)
        if ck.get("format")!="driveretriever-stage2b-vision-v1":
            raise RuntimeError(f"Unexpected Stage2B checkpoint: {ck.get('format')}")
        self.ck=ck
        self.device=torch.device(device if device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        meta=ck["cache_meta"]; self.model_name=meta["model_name"]
        self.max_length=int(meta.get("max_text_length",max_length))
        root=resolve_model(model_path or self.model_name,cache_dir,revision,local_files_only)
        self.identity=model_identity(self.model_name,root)
        self.tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
        td=torch.float16 if dtype=="fp16" and self.device.type=="cuda" else torch.float32
        self.text_only=bool(text_only)
        if self.text_only:
            from transformers import AutoConfig, SiglipTextModel
            config=AutoConfig.from_pretrained(root,local_files_only=True)
            if config.model_type != "siglip":
                raise RuntimeError("Text-only SigLIP loading requires model_type=siglip; use --siglip-full-model")
            self.backbone,info=SiglipTextModel.from_pretrained(root,dtype=td,local_files_only=True,output_loading_info=True)
            unexpected=[k for k in info.get("unexpected_keys",[]) if not k.startswith("vision_model.") and k not in {"logit_scale","logit_bias"}]
            if unexpected or any(info.get(k) for k in ("missing_keys","mismatched_keys","error_msgs")):
                raise RuntimeError(f"Unsafe text-only SigLIP weight mapping: {info}; use --siglip-full-model")
            self.backbone=self.backbone.to(self.device).eval()
        else:
            self.backbone=AutoModel.from_pretrained(root,torch_dtype=td,local_files_only=True).to(self.device).eval()
        self.text_proj=make_projection(ck["head_state_dict"],"text_proj").to(self.device).eval()
        for p in self.backbone.parameters(): p.requires_grad_(False)
        for p in self.text_proj.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self,texts,batch_size=128):
        texts=list(texts); raw_all=[]; proj_all=[]
        for s in range(0,len(texts),batch_size):
            x=self.tok(texts[s:s+batch_size],padding="max_length",truncation=True,
                       max_length=self.max_length,return_tensors="pt",return_attention_mask=True)
            ids=x["input_ids"].to(self.device); mask=x["attention_mask"].to(self.device)
            features=self.backbone(input_ids=ids,attention_mask=mask) if self.text_only else self.backbone.get_text_features(input_ids=ids,attention_mask=mask)
            y=get_tensor(features).float()
            raw=F.normalize(y,dim=-1)
            proj=F.normalize(self.text_proj(y),dim=-1)
            raw_all.append(raw.cpu().numpy()); proj_all.append(proj.cpu().numpy())
        if not raw_all:
            hd=int(self.ck["cache_meta"]["hidden_dim"])
            ed=int(self.ck["head_state_dict"]["text_proj.1.weight"].shape[0])
            return np.zeros((0,hd),np.float32),np.zeros((0,ed),np.float32)
        return np.concatenate(raw_all).astype(np.float32),np.concatenate(proj_all).astype(np.float32)
