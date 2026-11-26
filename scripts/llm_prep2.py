import os
ROOT=os.environ.get('MP_ROOT',os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys,os,json,numpy as np,onnxruntime as ort,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import llm_run as R
D=ROOT+'/llm/'; C=128; PY=sys.executable; os.makedirs(D+'calib',exist_ok=True)
s=ort.InferenceSession(D+'smol_c128.onnx',providers=['CPUExecutionProvider'])
def step(x,cos,sin,mask,pk,pv): return s.run(None,{'x':x,'cos':cos,'sin':sin,'mask':mask,'past_k':pk,'past_v':pv})
pre=R.make_prefix(step,C); P=len(pre[0]); print('prefix tokens',P)
with open(D+'prefix.bin','wb') as f:
    f.write(np.array([P],np.int32).tobytes()); f.write(np.ascontiguousarray(pre[1][:,:,:P]).tobytes()); f.write(np.ascontiguousarray(pre[2][:,:,:P]).tobytes())
rec=[]
def rstep(x,cos,sin,mask,pk,pv):
    o=step(x,cos,sin,mask,pk,pv); rec.append((x.copy(),cos.copy(),sin.copy(),mask.copy(),pk.copy(),pv.copy())); return o
prompts=["What is the capital of France?","Name three primary colors.","Write a short poem about the sea.","Explain what a robot is in one sentence.","How many legs does a spider have?","Who wrote Romeo and Juliet?","What is 12 plus 30?","Give me a tip for studying.","Why is the sky blue?","Translate hello to Spanish."]
for p in prompts:
    t,ni,no,dt,_=R.generate(rstep,p,C,24,prefix=pre); print(repr(p),'->',repr(t))
print(len(rec),'steps'); idx=np.linspace(0,len(rec)-1,47).astype(int)
names=['x','cos','sin','mask','pk','pv']
for k,nm in enumerate(names):
    a=np.stack([rec[i][k][0] for i in idx]).astype(np.float32); np.save(D+f'calib/all_{nm}.npy',a)
# stats for rmsnorm scaling
subprocess.run([PY,ROOT+'/scripts/llm_build.py','128',D+'smol_stats.onnx','stats'],check=True,stdout=subprocess.DEVNULL)
st=ort.InferenceSession(D+'smol_stats.onnx',providers=['CPUExecutionProvider']); ms=[];mx=[]
for i in idx:
    x,cos,sin,mask,pk,pv=rec[i]; o=st.run(None,{'x':x,'cos':cos,'sin':sin,'mask':mask,'past_k':pk,'past_v':pv})
    ms.append([float(v.ravel()[0]) for v in o[3::2]]); mx.append([float(v.ravel()[0]) for v in o[4::2]])
ms=np.array(ms); mx=np.array(mx); med=np.sqrt(np.median(ms,0))
sc=[float(2.0**np.round(-np.log2(m))) for m in med]; json.dump(sc,open(D+'rms_scales.json','w')); print('scales min/max',min(sc),max(sc))
print('scaled rms range per site (min..max over steps) worst:',float((np.sqrt(ms)*np.array(sc)).max()),float((np.sqrt(ms)*np.array(sc)).min()),'scaled absmax worst',float((mx*np.array(sc)).max()))
subprocess.run([PY,ROOT+'/scripts/llm_build.py','128',D+'smol_s.onnx',D+'rms_scales.json'],check=True)
