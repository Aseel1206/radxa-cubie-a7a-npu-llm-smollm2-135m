source /workspace/mp/scripts/env.sh
mkdir -p /workspace/mp/llm/c128s; cd /workspace/mp/llm/c128s
$PEG import onnx --model ../smol_s.onnx --output-model net.json --output-data net.data 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^E |rror|Traceback|not support|Unsupport|Warning\(" | head -20
$PEG generate inputmeta --model net.json --input-meta-output net_inputmeta.yml 2>&1 | tail -2
ls -la; chmod -R a+rwX .
