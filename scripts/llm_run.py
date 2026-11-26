import os
ROOT=os.environ.get('MP_ROOT',os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys,time,numpy as np
from tokenizers import Tokenizer
import os; D=os.environ.get('LLM_DIR',ROOT+'/llm/')
tok=Tokenizer.from_file(D+'tokenizer.json'); E=np.memmap(D+'embed.f32',dtype=np.float32,mode='r').reshape(-1,576)
MASKV=-30.0
L3,HD=90,64; INV=1.0/(100000**(np.arange(0,HD,2)/HD))
def rope(pos): f=pos*INV; e=np.concatenate([f,f]); return np.cos(e)[None].astype(np.float32),np.sin(e)[None].astype(np.float32)
PREFIX="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
def make_prefix(step,C):
    ids=tok.encode(PREFIX).ids; pk=np.zeros((1,L3,C-1,HD),np.float32); pv=np.zeros_like(pk)
    for pos,cur in enumerate(ids):
        x=np.array(E[cur],np.float32)[None]; cos,sin=rope(pos); mask=np.full((1,1,1,C),MASKV,np.float32); mask[...,:pos]=0; mask[...,C-1]=0
        _,nk,nv=step(x,cos,sin,mask,pk,pv); pk[:,:,pos]=nk[:,:,0]; pv[:,:,pos]=nv[:,:,0]
    return ids,pk,pv
def generate(step,prompt,C,maxnew=40,verbose=True,prefix=None):
    suf=tok.encode(f"{prompt}<|im_end|>\n<|im_start|>assistant\n").ids
    if prefix: pids,pk,pv=prefix; pk=pk.copy(); pv=pv.copy(); ids=pids+suf; pos=len(pids); cur=ids[pos]
    else:
        ids=tok.encode(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n").ids
        pk=np.zeros((1,L3,C-1,HD),np.float32); pv=np.zeros_like(pk); pos=0; cur=ids[0]
    out=[]; t0=time.time(); tp=None; lg_all=[]
    while pos<C-1 and len(out)<maxnew:
        x=np.array(E[cur],np.float32)[None]; cos,sin=rope(pos)
        mask=np.full((1,1,1,C),MASKV,np.float32); mask[...,:pos]=0; mask[...,C-1]=0
        lg,nk,nv=step(x,cos,sin,mask,pk,pv); pk[:,:,pos]=nk[:,:,0]; pv[:,:,pos]=nv[:,:,0]; lg_all.append(lg.ravel()); pos+=1
        if pos<len(ids): cur=ids[pos]
        else:
            if tp is None: tp=time.time()
            cur=int(lg.argmax())
            if cur in (2,0) or cur==tok.token_to_id('<|im_end|>'): break
            out.append(cur)
    td=time.time()-(tp or t0)
    txt=tok.decode(out)
    return txt,len(ids),len(out),(td/max(len(out),1)),lg_all
if __name__=='__main__':
    import onnxruntime as ort
    C=int(sys.argv[2]); s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider'])
    def step(x,cos,sin,mask,pk,pv): return s.run(None,{'x':x,'cos':cos,'sin':sin,'mask':mask,'past_k':pk,'past_v':pv})
    for p in ["What is the capital of France?","Name three primary colors."]:
        t,ni,no,dt,_=generate(step,p,C,30); print(repr(p),'->',repr(t),f'| prompt {ni} tok, {no} new, {1/dt:.1f} tok/s (laptop CPU)')
