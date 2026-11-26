import sys,os,time,numpy as np,onnxruntime as ort
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import llm_run as R
th=int(sys.argv[2]); o=ort.SessionOptions(); o.intra_op_num_threads=th; o.inter_op_num_threads=1; o.log_severity_level=3
s=ort.InferenceSession(sys.argv[1],o,providers=['CPUExecutionProvider'])
def step(x,cos,sin,mask,pk,pv): return s.run(None,{'x':x,'cos':cos,'sin':sin,'mask':mask,'past_k':pk,'past_v':pv})
t,ni,no,dt,_=R.generate(step,"Why is the sky blue?",128,30)
print(f'CPU fp32 threads={th}: {t!r}  decode {1e3*dt:.0f} ms/tok = {1/dt:.2f} tok/s')
