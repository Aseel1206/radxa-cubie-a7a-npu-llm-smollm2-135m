#!/usr/bin/env python3
"""Interactive chat with SmolLM2-135M-Instruct running on the Radxa NPU (int16 .nb). Commands: /reset /temp T /greedy /quit"""
import subprocess,sys,os,time
from tokenizers import Tokenizer
H=os.path.dirname(os.path.abspath(__file__)); os.environ['LD_LIBRARY_PATH']=os.path.expanduser('~/awnn_inference/install/lib')
tok=Tokenizer.from_file(H+'/tokenizer.json')
import json
NB=H+'/smollm2_135m_c128_int16.nb'   # fixed-point scale (fl) of every input is read from the metadata file Acuity wrote next to the model
meta=json.load(open(H+'/smollm2_135m_c128_int16.nbg_meta.json'))['Inputs']
FL=','.join(str((v.get('quantize') or {}).get('fl',0)) for v in meta.values())   # x,cos,sin,mask(fp16 -> 0),past_k,past_v
p=subprocess.Popen([H+'/nb_chat','-m',NB,'-e',H+'/embed.f32','-pre',H+'/prefix.bin','-fl',FL],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
def rl():
    while True:
        l=p.stdout.readline()
        if not l: sys.exit('engine died')
        if l.startswith(('READY','OK','TOK','DONE','ERR')): return l.strip()
def send(c): p.stdin.write(c+'\n'); p.stdin.flush()
print('loading model on the NPU...'); r=rl(); assert r.startswith('READY'),r; _,P,CTX=r.split(); CTX=int(CTX)
print(f'ready. context {CTX} tokens ({P} used by the fixed system prompt). Type a message; /reset, /temp 0.7, /greedy, /quit.\n')
send('TEMP 0.7 40 1.1'); rl(); turn=0; maxnew=120
while True:
    try: u=input('you> ').strip()
    except (EOFError,KeyboardInterrupt): break
    if not u: continue
    if u=='/quit': break
    if u=='/reset': send('RESET'); rl(); turn=0; print('[conversation reset]'); continue
    if u=='/greedy': send('TEMP 0 40 1.1'); rl(); print('[greedy decoding]'); continue
    if u.startswith('/temp'): send(f'TEMP {float(u.split()[1])} 40 1.1'); rl(); print('[sampling]'); continue
    text=(u if turn==0 else '<|im_end|>\n<|im_start|>user\n'+u)+'<|im_end|>\n<|im_start|>assistant\n'
    ids=tok.encode(text).ids; send('GEN %d %s'%(maxnew,' '.join(map(str,ids)))); out=[]; shown=''; print('bot> ',end='',flush=True)
    while True:
        r=rl()
        if r.startswith('TOK'):
            out.append(int(r.split()[1])); t=tok.decode(out)
            if not t.endswith('�'): print(t[len(shown):],end='',flush=True); shown=t
        elif r.startswith('DONE'):
            f=r.split(); n,mspt,pos,flag=int(f[1]),float(f[2]),int(f[3]),int(f[4]); break
        else: sys.exit(r)
    print(); 
    if flag==2: print('[context full - resetting the conversation]'); send('RESET'); rl(); turn=0
    else: turn+=1
    print(f'   [{n} tokens, {1000/mspt:.1f} tok/s on NPU, context {pos}/{CTX}]' if n else '   [no reply]')
send('QUIT')
