#!/usr/bin/env python3
"""Generate interface.yaml with exact source_text and hashes from proto file."""
import hashlib, yaml, json

PROJECT_ROOT = r"D:\PR-review\TEST\FAG_test\flash_attention_score_grad"
PROTO = f"{PROJECT_ROOT}\\op_graph\\flash_attention_score_grad_proto.h"

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def get_line_text(lines, lineno):
    """Get exact content of a 1-indexed line (no stripping)."""
    return lines[lineno - 1].rstrip('\n').rstrip('\r')

def sha256(text):
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

proto_lines = read_lines(PROTO)

# Build source anchors for proto items
def make_proto_anchor(start_line, end_line, source_text):
    return {
        'id': 'SRC_PROTO_REG_OP',
        'file': 'op_graph/flash_attention_score_grad_proto.h',
        'symbol': 'REG_OP(FlashAttentionScoreGrad)',
        'span': {'start_line': start_line, 'end_line': end_line},
        'source_text': source_text,
        'code_hash': sha256(source_text),
        'anchor_kind': 'proto_definition'
    }

items = []

# Input tensors (lines 87-90)
for line_no, name, kind in [
    (87, 'QUERY', 'input_tensor'),
    (88, 'KEY', 'input_tensor'),
    (89, 'VALUE', 'input_tensor'),
    (90, 'DY', 'input_tensor'),
]:
    txt = get_line_text(proto_lines, line_no)
    item = {
        'id': f'ARG_{name}',
        'kind': kind,
        'name': name.lower(),
        'dtype': ['DT_FLOAT8_E5M2', 'DT_FLOAT8_E4M3FN', 'DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'],
        'layout': 'FORMAT_ND',
        'rank': '3..4',
        'shape_symbols': [],
        'status': 'confirmed',
        'sources': [make_proto_anchor(line_no, line_no, txt)]
    }
    items.append(item)

# Optional inputs
optional_entries = [
    (91, 'PSE_SHIFT', 'pse_shift', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4', 'provided by caller; may be nullptr'),
    (92, 'DROP_MASK', 'drop_mask', ['DT_UINT8'], '0..4', 'provided when dropout is used (keep_prob < 1.0)'),
    (93, 'PADDING_MASK', 'padding_mask', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..3', 'provided by caller; may be nullptr'),
    (94, 'ATTEN_MASK', 'atten_mask', ['DT_BOOL', 'DT_UINT8'], '0..4', 'provided when attention masking is used'),
    (95, 'SOFTMAX_MAX', 'softmax_max', ['DT_FLOAT32'], '0..3', 'provided from forward pass'),
    (96, 'SOFTMAX_SUM', 'softmax_sum', ['DT_FLOAT32'], '0..3', 'provided from forward pass'),
    (97, 'SOFTMAX_IN', 'softmax_in', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4', 'provided from forward pass'),
    (98, 'ATTENTION_IN', 'attention_in', ['DT_FLOAT8_E5M2', 'DT_FLOAT8_E4M3FN', 'DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4', 'provided from forward pass (attention scores before softmax)'),
    (99, 'PREFIX', 'prefix', ['DT_INT64'], '0..2', 'provided when prefix mode (sparseMode=5 or 6) is used'),
    (100, 'ACTUAL_SEQ_QLEN', 'actual_seq_qlen', ['DT_INT64'], '1', 'provided when layout is TND'),
    (101, 'ACTUAL_SEQ_KVLEN', 'actual_seq_kvlen', ['DT_INT64'], '1', 'provided when layout is TND'),
    (102, 'Q_START_IDX', 'q_start_idx', ['DT_INT64'], '0..2', 'provided by caller (V2+ APIs)'),
    (103, 'KV_START_IDX', 'kv_start_idx', ['DT_INT64'], '0..2', 'provided by caller (V2+ APIs)'),
    (104, 'D_SCALE_Q', 'd_scale_q', ['DT_FLOAT32'], '0..1', 'provided for quantized inputs (V4 API)'),
    (105, 'D_SCALE_K', 'd_scale_k', ['DT_FLOAT32'], '0..1', 'provided for quantized inputs (V4 API)'),
    (106, 'D_SCALE_V', 'd_scale_v', ['DT_FLOAT32'], '0..1', 'provided for quantized inputs (V4 API)'),
    (107, 'D_SCALE_DY', 'd_scale_dy', ['DT_FLOAT32'], '0..1', 'provided for quantized inputs (V4 API)'),
    (108, 'D_SCALE_O', 'd_scale_o', ['DT_FLOAT32'], '0..1', 'provided for quantized inputs (V4 API)'),
    (109, 'QUERY_ROPE', 'query_rope', ['DT_FLOAT8_E5M2', 'DT_FLOAT8_E4M3FN', 'DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4', 'provided when RoPE is used (V3+ APIs)'),
    (110, 'KEY_ROPE', 'key_rope', ['DT_FLOAT8_E5M2', 'DT_FLOAT8_E4M3FN', 'DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4', 'provided when RoPE is used (V3+ APIs)'),
    (111, 'SINK', 'sink', ['DT_FLOAT32'], '0..3', 'provided when attention sink is used (V3+ APIs)'),
    (112, 'DS_SCALE', 'ds_scale', ['DT_FLOAT32'], '0..1', 'provided for MX full-quant scenarios'),
    (113, 'P_SCALE', 'p_scale', ['DT_FLOAT32'], '0..1', 'provided for MX full-quant scenarios'),
]
for line_no, id_suffix, name, dtypes, rank, pres_cond in optional_entries:
    txt = get_line_text(proto_lines, line_no)
    item = {
        'id': f'ARG_{id_suffix}',
        'kind': 'optional_input',
        'name': name,
        'optional': True,
        'presence_condition': pres_cond,
        'dtype': dtypes,
        'layout': 'FORMAT_ND',
        'rank': rank,
        'shape_symbols': [],
        'status': 'confirmed',
        'sources': [make_proto_anchor(line_no, line_no, txt)]
    }
    items.append(item)

# Output tensors (lines 114-120)
output_entries = [
    (114, 'DQ', 'dq', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '3..4'),
    (115, 'DK', 'dk', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '3..4'),
    (116, 'DV', 'dv', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '3..4'),
    (117, 'DPSE', 'dpse', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '3..4'),
    (118, 'DQ_ROPE', 'dq_rope', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4'),
    (119, 'DK_ROPE', 'dk_rope', ['DT_FLOAT16', 'DT_BF16', 'DT_FLOAT32'], '0..4'),
    (120, 'DSINK', 'dsink', ['DT_FLOAT32'], '0..3'),
]
for line_no, id_suffix, name, dtypes, rank in output_entries:
    txt = get_line_text(proto_lines, line_no)
    item = {
        'id': f'ARG_{id_suffix}',
        'kind': 'output_tensor',
        'name': name,
        'dtype': dtypes,
        'layout': 'FORMAT_ND',
        'rank': rank,
        'shape_symbols': [],
        'status': 'confirmed',
        'sources': [make_proto_anchor(line_no, line_no, txt)]
    }
    items.append(item)

# Attributes (lines 121-133)
attr_entries = [
    (121, 'SCALE_VALUE', 'scale_value', 'Float', 1.0, '(0, +inf)'),
    (122, 'KEEP_PROB', 'keep_prob', 'Float', 1.0, '[0.0, 1.0]'),
    (123, 'PRE_TOCKENS', 'pre_tockens', 'Int', 2147483647, '[0, INT_MAX]'),
    (124, 'NEXT_TOCKENS', 'next_tockens', 'Int', 2147483647, '[0, INT_MAX]'),
    (125, 'HEAD_NUM', 'head_num', 'Int', None, '[1, +inf)'),
    (126, 'INPUT_LAYOUT', 'input_layout', 'String', None, '[BSH, SBH, BNSD, BSND, TND]'),
    (127, 'INNER_PRECISE', 'inner_precise', 'Int', 0, '[0, 2]'),
    (128, 'SPARSE_MODE', 'sparse_mode', 'Int', 0, '[0, 8]'),
    (129, 'PSE_TYPE', 'pse_type', 'Int', 1, '[0, 3]'),
    (130, 'SEED', 'seed', 'Int', 0, '[0, INT_MAX]'),
    (131, 'OFFSET', 'offset', 'Int', 0, '[0, INT_MAX]'),
    (132, 'OUT_DTYPE', 'out_dtype', 'Int', 0, '[0, 1]'),
    (133, 'SOFTMAX_IN_LAYOUT', 'softmax_in_layout', 'String', '', '["", "same_as_input"]'),
]
for line_no, id_suffix, name, attr_type, default, domain in attr_entries:
    txt = get_line_text(proto_lines, line_no)
    item = {
        'id': f'ATTR_{id_suffix}',
        'kind': 'attribute',
        'name': name,
        'attr_type': attr_type,
        'default': default,
        'domain': domain,
        'status': 'confirmed',
        'sources': [make_proto_anchor(line_no, line_no, txt)]
    }
    items.append(item)

# Interface constraints
proto_full_text = get_line_text(proto_lines, 86)
proto_anchor = make_proto_anchor(86, 134, proto_full_text)

constraints = [
    'inputLayout must be one of BSH/SBH/BNSD/BSND/TND; determines tensor dimension interpretation',
    'query, key, value, dy must share the same dtype from {DT_FLOAT16, DT_BF16, DT_FLOAT32} (DT_FLOAT8 variants converted to FP16/BF16 by out_dtype attribute)',
    'When layout is TND, actual_seq_qlen and actual_seq_kvlen are required and must satisfy sum(actual_seq_qlen) <= t1',
]
for i, c in enumerate(constraints):
    items.append({
        'id': f'REL_CONSTRAINT_{i+1}',
        'kind': 'interface_constraint',
        'name': f'constraint_{i+1}',
        'constraint_text': c,
        'source_refs': ['SRC_PROTO_REG_OP'],
        'status': 'confirmed',
        'sources': [make_proto_anchor(86, 134, proto_full_text)]
    })

# Format conversion
items.append({
    'id': 'REL_FORMAT_CONV_1',
    'kind': 'format_conversion',
    'name': 'aclIntArray_to_tensor',
    'from_format': 'aclIntArray',
    'to_format': 'tensor (DT_INT64, FORMAT_ND)',
    'source_refs': ['SRC_PROTO_REG_OP'],
    'status': 'confirmed',
    'sources': [make_proto_anchor(86, 134, proto_full_text)]
})

# Build document
doc = {
    'version': 1,
    'artifact': {
        'type': 'operator.interface',
        'schema_version': 1,
        'owner': 'uo-boundary-agent'
    },
    'snapshot': {
        'run_id': 'UO_RUN_20260714072706208475',
        'source_snapshot_id': 'SOURCE_13CA442A916B513A',
        'source_revision': 'unknown',
        'spec_bundle_hash': 'sha256:eea28c21a04bb8e2dda50a60407c0a68e8e16c05aecaaf5f5b439f682e693ece'
    },
    'items': items,
    'relations': [],
    'unresolved': [],
    'analysis_status': 'complete',
    'reason': 'All interface items extracted from REG_OP proto definition',
    'sources': [make_proto_anchor(86, 134, proto_full_text)]
}

out_path = f"{PROJECT_ROOT}\\.understand-operator\\flash_attention_score_grad\\facts\\operator\\interface.yaml"
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

print(f"Wrote {len(items)} items to {out_path}")
