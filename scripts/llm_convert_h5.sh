# usage: llm_convert.sh <c128> <qtype: i16|pcq|u8>
source /workspace/mp/scripts/env.sh
D=/workspace/mp/llm/$1; q=$2; cd $D
python3 - <<'PY'
import yaml
m=yaml.safe_load(open('net_inputmeta.yml'))
db=m['input_meta']['databases'][0]; names=['x','cos','sin','mask','pk','pv']
new=[]
for nm,p in zip(names,db['ports']):
    p=dict(p); p['category']='undefined'
    new.append({'path':f'/workspace/mp/llm/calib/all_{nm}.npy','type':'NPY','ports':[p]})
m['input_meta']['databases']=[{'path':'/workspace/mp/llm/calib/all.h5','type':'H5FS','ports':[dict(p,tensor_name=nm,category='undefined') for nm,p in zip(names,db['ports'])]}]
[pp['preprocess'].__setitem__('reverse_channel',False) for pp in m['input_meta']['databases'][0]['ports']]
yaml.safe_dump(m,open("net_inputmeta_h5.yml","w"),sort_keys=False)
print('ports',[d['ports'][0]['lid'] for d in new])
PY
$PEG generate postprocess-file --model net.json --postprocess-file-output net_postprocess_file.yml >/dev/null 2>&1
sed -i 's/add_postproc_node: false/add_postproc_node: true/' net_postprocess_file.yml
case $q in pcq) a="perchannel_symmetric_affine int8 pcq";; i16) a="dynamic_fixed_point int16 i16";; u8) a="asymmetric_affine uint8 u8";; esac
set -- $a
echo "== quantize $3"; date
$PEG quantize --model net.json --model-data net.data --device CPU --with-input-meta net_inputmeta_h5.yml --iterations 47 --rebuild --model-quantize net_$3.quantize --quantizer $1 --qtype $2 > quant_$3.log 2>&1
tail -3 quant_$3.log | sed 's/\x1b\[[0-9;]*m//g'; date
echo "== export $3"
$PEG export ovxlib --model net.json --model-data net.data --model-quantize net_$3.quantize --with-input-meta net_inputmeta_h5.yml \
  --postprocess-file net_postprocess_file.yml --dtype quantized --pack-nbg-unify --optimize VIP9000NANODI_PLUS_PID0X1000003B \
  --viv-sdk $VIVANTE_SDK_DIR --output-path $D/out_$3/net > export_$3.log 2>&1
ls -la $D/out_$3_nbg_unify/network_binary.nb 2>/dev/null || { echo "FAILED $3"; sed 's/\x1b\[[0-9;]*m//g' export_$3.log | grep -E "^E |Fatal|rror" | head -8; }
date; chmod -R a+rwX $D
