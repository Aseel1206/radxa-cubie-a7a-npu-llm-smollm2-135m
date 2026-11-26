"""Build a static single-token decode graph of Llama-style SmolLM2-135M from safetensors, plain ONNX ops only.
inputs : x (1,576) token embedding | cos,sin (1,64) | mask (1,1,1,C) additive | past_k, past_v (1,90,C-1,64)
outputs: logits (1,49152) | new_k, new_v (1,90,1,64)"""
import os
ROOT=os.environ.get('MP_ROOT',os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys,struct,json,numpy as np,onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
C=int(sys.argv[1]); OUT=sys.argv[2]; STATS=len(sys.argv)>3 and sys.argv[3]=='stats'; SCALES=json.load(open(sys.argv[3])) if len(sys.argv)>3 and not STATS else None; sites=[]; D=ROOT+'/llm/'
f=open(D+'model.safetensors','rb'); n=struct.unpack('<Q',f.read(8))[0]; hd=json.loads(f.read(n)); base=8+n
def W(k):
    m=hd[k]; f.seek(base+m['data_offsets'][0]); b=np.frombuffer(f.read(m['data_offsets'][1]-m['data_offsets'][0]),np.uint16)
    return (b.astype(np.uint32)<<16).view(np.float32).reshape(m['shape'])
L,NH,NKV,HD,DM,FF,V,EPS=30,9,3,64,576,1536,49152,1e-5
nodes=[];init=[]
def I(name,a): init.append(nh.from_array(np.asarray(a),name)); return name
cnt=[0]
def N(op,ins,outn=None,**kw):
    cnt[0]+=1; o=outn or f'{op}_{cnt[0]}'; nodes.append(h.make_node(op,ins,[o],name=o+'_n',**kw)); return o
c=lambda v,dt=np.float32: I(f'c{len(init)}',np.array(v,dt))
def rms(x,w,p):
    sites.append(x)
    if SCALES:
        sc=SCALES[len(sites)-1]; x=N('Mul',[x,c([sc])]); x2=N('Mul',[x,x]); m=N('ReduceMean',[x2],axes=[1],keepdims=1); e=N('Add',[m,c([EPS*sc*sc])]); r=N('Sqrt',[e]); d=N('Div',[x,r]); return N('Mul',[d,I(p,w)])
    x2=N('Mul',[x,x]); m=N('ReduceMean',[x2],axes=[1],keepdims=1); e=N('Add',[m,c([EPS])]); r=N('Sqrt',[e]); d=N('Div',[x,r]); return N('Mul',[d,I(p,w)])
def sl(t,starts,ends,axes): return N('Slice',[t,I(f's{len(init)}',np.array(starts,np.int64)),I(f'e{len(init)}',np.array(ends,np.int64)),I(f'a{len(init)}',np.array(axes,np.int64))])
def rope(t,cos4,sin4):
    a=sl(t,[0],[32],[3]); b=sl(t,[32],[64],[3]); nb=N('Mul',[b,c([-1.0])]); rh=N('Concat',[nb,a],axis=3)
    return N('Add',[N('Mul',[t,cos4]),N('Mul',[rh,sin4])])
shp=lambda s: I(f'sh{len(init)}',np.array(s,np.int64))
x='x'; cos4=N('Reshape',['cos',shp([1,1,1,HD])]); sin4=N('Reshape',['sin',shp([1,1,1,HD])])
nk=[];nv=[]
for l in range(L):
    p=f'model.layers.{l}.'
    hN=rms(x,W(p+'input_layernorm.weight'),f'ln1_{l}')
    q=N('MatMul',[hN,I(f'wq{l}',W(p+'self_attn.q_proj.weight').T.copy())]); k=N('MatMul',[hN,I(f'wk{l}',W(p+'self_attn.k_proj.weight').T.copy())]); v=N('MatMul',[hN,I(f'wv{l}',W(p+'self_attn.v_proj.weight').T.copy())])
    q=rope(N('Reshape',[q,shp([1,NH,1,HD])]),cos4,sin4); k=rope(N('Reshape',[k,shp([1,NKV,1,HD])]),cos4,sin4); v=N('Reshape',[v,shp([1,NKV,1,HD])])
    nk.append(k); nv.append(v)
    pk=sl('past_k',[NKV*l],[NKV*l+NKV],[1]); pv=sl('past_v',[NKV*l],[NKV*l+NKV],[1])
    K=N('Concat',[pk,k],axis=2); Vv=N('Concat',[pv,v],axis=2)
    qg=N('Reshape',[q,shp([1,NKV,NH//NKV,HD])]); Kt=N('Transpose',[K],perm=[0,1,3,2])
    s=N('MatMul',[qg,Kt]); s=N('Mul',[s,c([1/8.0])]); s=N('Add',[s,'mask']); pr=N('Softmax',[s],axis=-1)
    o=N('MatMul',[pr,Vv]); o=N('Reshape',[o,shp([1,DM])]); o=N('MatMul',[o,I(f'wo{l}',W(p+'self_attn.o_proj.weight').T.copy())]); x=N('Add',[x,o])
    h2=rms(x,W(p+'post_attention_layernorm.weight'),f'ln2_{l}')
    g=N('MatMul',[h2,I(f'wg{l}',W(p+'mlp.gate_proj.weight').T.copy())]); u=N('MatMul',[h2,I(f'wu{l}',W(p+'mlp.up_proj.weight').T.copy())])
    sg=N('Mul',[g,N('Sigmoid',[g])]); d=N('MatMul',[N('Mul',[sg,u]),I(f'wd{l}',W(p+'mlp.down_proj.weight').T.copy())]); x=N('Add',[x,d])
x=rms(x,W('model.norm.weight'),'lnf')
N('MatMul',[x,I('lm_head',W('model.embed_tokens.weight').T.copy())],'logits')
N('Concat',nk,'new_k',axis=1); N('Concat',nv,'new_v',axis=1)
vi=lambda n,s: h.make_tensor_value_info(n,T.FLOAT,s)
extra=[]
if STATS:
    for i,t in enumerate(sites):
        o=N('ReduceMean',[N('Mul',[t,t])],f'ms_{i}',axes=[1],keepdims=1); extra.append(vi(o,[1,1]))
        a=N('ReduceMax',[N('Abs',[t])],f'mx_{i}',axes=[1],keepdims=1); extra.append(vi(a,[1,1]))
g=h.make_graph(nodes,'smollm2',[vi('x',[1,DM]),vi('cos',[1,HD]),vi('sin',[1,HD]),vi('mask',[1,1,1,C]),vi('past_k',[1,L*NKV,C-1,HD]),vi('past_v',[1,L*NKV,C-1,HD])],
  [vi('logits',[1,V]),vi('new_k',[1,L*NKV,1,HD]),vi('new_v',[1,L*NKV,1,HD])]+extra,init)
m=h.make_model(g,opset_imports=[h.make_opsetid('',13)]); m.ir_version=8; onnx.checker.check_model(m,full_check=False)
onnx.save(m,OUT,save_as_external_data=False); print('saved',OUT,len(nodes),'nodes')
# embedding table for CPU lookup (fp32 npy) 
np.save(D+'embed.npy',W('model.embed_tokens.weight').astype(np.float32))
W('model.embed_tokens.weight').astype(np.float32).tofile(D+'embed.f32')   # raw fp32 table for the board-side embedding lookup
