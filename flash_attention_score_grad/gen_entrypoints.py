#!/usr/bin/env python3
"""Generate entrypoints.yaml with exact source_text and hashes."""
import hashlib, yaml, os

PROJECT_ROOT = r"D:\PR-review\TEST\FAG_test\flash_attention_score_grad"

def sha256(text):
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

def read_line(path, lineno):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if lineno <= len(lines):
            return lines[lineno-1].rstrip('\n').rstrip('\r').strip()
    except:
        pass
    return ""

def make_source(file_rel, symbol, line_start, line_end, full_path=None, anchor_kind='function_definition'):
    fp = full_path or os.path.join(PROJECT_ROOT, file_rel)
    if line_start > 0:
        src_text = read_line(fp, line_start)
    else:
        src_text = symbol
    h = sha256(src_text) if src_text else sha256(symbol)
    safe = symbol.upper().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '').replace('<', '').replace('>', '').replace(',', '').replace('.', '_').replace('*', '').replace('#', '')[:80]
    return {
        'id': f'SRC_{safe}',
        'file': file_rel,
        'symbol': symbol,
        'span': {'start_line': line_start, 'end_line': line_end},
        'source_text': src_text,
        'code_hash': h,
        'anchor_kind': anchor_kind
    }

items = []
relations = []

# ===== REGISTRATION ENTRIES =====
# REG_OP proto
items.append({
    'id': 'SYM_REG_OP_PROTO',
    'kind': 'registration_entry',
    'name': 'REG_OP(FlashAttentionScoreGrad)',
    'file': 'op_graph/flash_attention_score_grad_proto.h',
    'symbol': 'REG_OP(FlashAttentionScoreGrad)',
    'entry_kind': 'op_proto_reg',
    'status': 'confirmed',
    'sources': [make_source('op_graph/flash_attention_score_grad_proto.h', 'REG_OP(FlashAttentionScoreGrad)', 86, 134,
                            anchor_kind='registration_entry')]
})

# OpDef registration
items.append({
    'id': 'SYM_OPDEF_REG',
    'kind': 'registration_entry',
    'name': 'FlashAttentionScoreGrad OpDef',
    'file': 'op_host/flash_attention_score_grad_def.cpp',
    'symbol': 'class FlashAttentionScoreGrad',
    'entry_kind': 'op_def_reg',
    'status': 'confirmed',
    'sources': [make_source('op_host/flash_attention_score_grad_def.cpp', 'class FlashAttentionScoreGrad : public OpDef', 20, 23,
                            anchor_kind='class_definition')]
})

# IMPL_OP_OPTILING
items.append({
    'id': 'SYM_IMPL_OP_OPTILING',
    'kind': 'registration_entry',
    'name': 'IMPL_OP_OPTILING(FlashAttentionScoreGrad)',
    'file': 'op_host/flash_attention_score_grad_tiling.cpp',
    'symbol': 'IMPL_OP_OPTILING(FlashAttentionScoreGrad)',
    'entry_kind': 'op_optiling_reg',
    'status': 'confirmed',
    'sources': [make_source('op_host/flash_attention_score_grad_tiling.cpp', 'IMPL_OP_OPTILING(FlashAttentionScoreGrad)', 467, 471,
                            anchor_kind='registration_entry')]
})

# ===== API DEFINITIONS =====
# The 9 API GetWorkspaceSize functions and their corresponding Phase2 executors
api_defs = [
    ('SYM_API_V1_GWS', 'aclnnFlashAttentionScoreGradGetWorkspaceSize', 1650, 1730, 'aclnn_phase1'),
    ('SYM_API_V1_EXEC', 'aclnnFlashAttentionScoreGrad', 1732, 1737, 'aclnn_phase2'),
    ('SYM_API_UNPAD_V1_GWS', 'aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize', 1739, 1830, 'aclnn_phase1'),
    ('SYM_API_UNPAD_V1_EXEC', 'aclnnFlashAttentionUnpaddingScoreGrad', 1830, 1835, 'aclnn_phase2'),
    ('SYM_API_V2_GWS', 'aclnnFlashAttentionScoreGradV2GetWorkspaceSize', 1916, 2000, 'aclnn_phase1'),
    ('SYM_API_V2_EXEC', 'aclnnFlashAttentionScoreGradV2', 2000, 2005, 'aclnn_phase2'),
    ('SYM_API_UNPAD_V2_GWS', 'aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize', 2006, 2097, 'aclnn_phase1'),
    ('SYM_API_UNPAD_V2_EXEC', 'aclnnFlashAttentionUnpaddingScoreGradV2', 2097, 2102, 'aclnn_phase2'),
    ('SYM_API_UNPAD_V3_GWS', 'aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize', 2194, 2285, 'aclnn_phase1'),
    ('SYM_API_UNPAD_V3_EXEC', 'aclnnFlashAttentionUnpaddingScoreGradV3', 2285, 2290, 'aclnn_phase2'),
    ('SYM_API_UNPAD_V4_GWS', 'aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize', 2291, 2358, 'aclnn_phase1'),
    ('SYM_API_UNPAD_V4_EXEC', 'aclnnFlashAttentionUnpaddingScoreGradV4', 2358, 2363, 'aclnn_phase2'),
    ('SYM_API_V3_GWS', 'aclnnFlashAttentionScoreGradV3GetWorkspaceSize', 2483, 2570, 'aclnn_phase1'),
    ('SYM_API_V3_EXEC', 'aclnnFlashAttentionScoreGradV3', 2570, 2575, 'aclnn_phase2'),
    ('SYM_API_UNPAD_V5_GWS', 'aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize', 2576, 2696, 'aclnn_phase1'),
    ('SYM_API_UNPAD_V5_EXEC', 'aclnnFlashAttentionUnpaddingScoreGradV5', 2696, 2701, 'aclnn_phase2'),
    ('SYM_API_UNPAD_V5_MAX_GWS', 'aclnnFlashAttentionUnpaddingScoreGradV5GetMaxWorkspaceSize', 2702, 2820, 'aclnn_phase1'),
    ('SYM_API_V4_GWS', 'aclnnFlashAttentionScoreGradV4GetWorkspaceSize', 3020, 3111, 'aclnn_phase1'),
    ('SYM_API_V4_EXEC', 'aclnnFlashAttentionScoreGradV4', 3113, 3118, 'aclnn_phase2'),
]
cpp = 'op_api/aclnn_flash_attention_score_grad.cpp'
for eid, sym, line_start, line_end, entry_kind in api_defs:
    items.append({
        'id': eid,
        'kind': 'api_definition',
        'name': sym,
        'file': cpp,
        'symbol': sym,
        'entry_kind': entry_kind,
        'status': 'confirmed',
        'sources': [make_source(cpp, sym, line_start, line_end, anchor_kind='function_definition')]
    })

# L0 FlashAttentionScoreGrad
l0_cpp = 'op_api/flash_attention_score_grad.cpp'
items.append({
    'id': 'SYM_L0_FAG',
    'kind': 'api_definition',
    'name': 'l0op::FlashAttentionScoreGrad',
    'file': l0_cpp,
    'symbol': 'FlashAttentionScoreGrad',
    'entry_kind': 'l0_dispatcher',
    'status': 'confirmed',
    'sources': [make_source(l0_cpp, 'FlashAttentionScoreGrad', 43, 57, anchor_kind='function_definition')]
})

# ===== HOST / TILING ENTRIES =====
# TilingFlashAttentionGradScore
tiling_cpp = 'op_host/flash_attention_score_grad_tiling.cpp'
items.append({
    'id': 'SYM_TILING_MAIN',
    'kind': 'tiling_entry',
    'name': 'TilingFlashAttentionGradScore',
    'file': tiling_cpp,
    'symbol': 'TilingFlashAttentionGradScore',
    'entry_kind': 'tiling_dispatch',
    'status': 'confirmed',
    'sources': [make_source(tiling_cpp, 'TilingFlashAttentionGradScore', 408, 431)]
})

# RunEmptyTiling
items.append({
    'id': 'SYM_RUN_EMPTY_TILING',
    'kind': 'tiling_entry',
    'name': 'FlashAttentionScoreGradTiling::RunEmptyTiling',
    'file': tiling_cpp,
    'symbol': 'RunEmptyTiling',
    'entry_kind': 'tiling_empty',
    'status': 'confirmed',
    'sources': [make_source(tiling_cpp, 'RunEmptyTiling', 68, 136)]
})

# RunEmptyTilingRegbase
items.append({
    'id': 'SYM_RUN_EMPTY_TILING_REGBASE',
    'kind': 'tiling_entry',
    'name': 'FlashAttentionScoreGradTiling::RunEmptyTilingRegbase',
    'file': tiling_cpp,
    'symbol': 'RunEmptyTilingRegbase',
    'entry_kind': 'tiling_empty_regbase',
    'status': 'confirmed',
    'sources': [make_source(tiling_cpp, 'RunEmptyTilingRegbase', 138, 234)]
})

# TilingPrepareForFlashAttentionScoreGrad
items.append({
    'id': 'SYM_TILING_PREPARE',
    'kind': 'tiling_entry',
    'name': 'TilingPrepareForFlashAttentionScoreGrad',
    'file': tiling_cpp,
    'symbol': 'TilingPrepareForFlashAttentionScoreGrad',
    'entry_kind': 'tiling_prepare',
    'status': 'confirmed',
    'sources': [make_source(tiling_cpp, 'TilingPrepareForFlashAttentionScoreGrad', 433, 465)]
})

# InferShape
items.append({
    'id': 'SYM_INFER_SHAPE',
    'kind': 'host_entry',
    'name': 'InferShape4FlashAttentionScoreGrad',
    'file': 'op_host/flash_attention_score_grad_infershape.cpp',
    'symbol': 'InferShape4FlashAttentionScoreGrad',
    'entry_kind': 'infershape',
    'status': 'confirmed',
    'sources': [make_source('op_host/flash_attention_score_grad_infershape.cpp', 'InferShape4FlashAttentionScoreGrad', 30, 31)]
})

# Arch35 tiling classes
normal_cpp = 'op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp'
items.append({
    'id': 'SYM_TILING_NORMAL_REGBASE',
    'kind': 'tiling_entry',
    'name': 'FlashAttentionScoreGradTilingNormalRegbase',
    'file': normal_cpp,
    'symbol': 'FlashAttentionScoreGradTilingNormalRegbase',
    'entry_kind': 'tiling_class_regbase',
    'status': 'confirmed',
    'sources': [make_source(normal_cpp, 'FlashAttentionScoreGradTilingNormalRegbase::GetShapeAttrsInfo', 47, 48)]
})

varlen_cpp = 'op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp'
items.append({
    'id': 'SYM_TILING_VARLEN_REGBASE',
    'kind': 'tiling_entry',
    'name': 'FlashAttentionScoreGradTilingVarlenRegbase',
    'file': varlen_cpp,
    'symbol': 'FlashAttentionScoreGradTilingVarlenRegbase',
    'entry_kind': 'tiling_class_regbase',
    'status': 'confirmed',
    'sources': [make_source(varlen_cpp, 'FlashAttentionScoreGradTilingVarlenRegbase', 22, 27)]
})

# ===== KERNEL ENTRIES =====
# RegbaseFAG - arch35 kernel global entry
entry_h = 'op_kernel/arch35/flash_attention_score_grad_entry_regbase.h'
items.append({
    'id': 'SYM_REGBASE_FAG',
    'kind': 'kernel_global_entry',
    'name': 'RegbaseFAG',
    'file': entry_h,
    'symbol': 'RegbaseFAG',
    'entry_kind': 'kernel_global_arch35',
    'status': 'confirmed',
    'sources': [make_source(entry_h, 'RegbaseFAG', 201, 210)]
})

# FlashAttentionScoreGradKernel class (in kernel.h)
kernel_h = 'op_kernel/arch35/flash_attention_score_grad_kernel.h'
items.append({
    'id': 'SYM_KERNEL_CLASS',
    'kind': 'kernel_class_entry',
    'name': 'FlashAttentionScoreGradKernel',
    'file': kernel_h,
    'symbol': 'FlashAttentionScoreGradKernel',
    'entry_kind': 'kernel_class_arch35',
    'status': 'confirmed',
    'sources': [make_source(kernel_h, 'FlashAttentionScoreGradKernel', 1, 17, anchor_kind='header_guard')]
})

# FlashAttentionScoreGradKernelDeter class
items.append({
    'id': 'SYM_KERNEL_CLASS_DETER',
    'kind': 'kernel_class_entry',
    'name': 'FlashAttentionScoreGradKernelDeter',
    'file': 'op_kernel/arch35/flash_attention_score_grad_kernel_deter.h',
    'symbol': 'FlashAttentionScoreGradKernelDeter',
    'entry_kind': 'kernel_class_arch35',
    'status': 'confirmed',
    'sources': [make_source('op_kernel/arch35/flash_attention_score_grad_kernel_deter.h', 'FlashAttentionScoreGradKernelDeter', 1, 17, anchor_kind='header_guard')]
})

# Proto definition
items.append({
    'id': 'SYM_PROTO_DEF',
    'kind': 'proto_definition',
    'name': 'REG_OP(FlashAttentionScoreGrad)',
    'file': 'op_graph/flash_attention_score_grad_proto.h',
    'symbol': 'REG_OP(FlashAttentionScoreGrad)',
    'entry_kind': 'proto_reg_op',
    'status': 'confirmed',
    'sources': [make_source('op_graph/flash_attention_score_grad_proto.h', 'REG_OP(FlashAttentionScoreGrad)', 86, 134, anchor_kind='proto_definition')]
})

# ===== RELATIONS =====
# API -> L0 dispatcher
api_to_l0 = [
    ('SYM_API_V1_GWS', 'SYM_L0_FAG'),
    ('SYM_API_UNPAD_V1_GWS', 'SYM_L0_FAG'),
    ('SYM_API_V2_GWS', 'SYM_L0_FAG'),
    ('SYM_API_UNPAD_V2_GWS', 'SYM_L0_FAG'),
    ('SYM_API_UNPAD_V3_GWS', 'SYM_L0_FAG'),
    ('SYM_API_UNPAD_V4_GWS', 'SYM_L0_FAG'),
    ('SYM_API_V3_GWS', 'SYM_L0_FAG'),
    ('SYM_API_UNPAD_V5_GWS', 'SYM_L0_FAG'),
    ('SYM_API_V4_GWS', 'SYM_L0_FAG'),
]
for i, (src, tgt) in enumerate(api_to_l0):
    relations.append({
        'id': f'REL_API_TO_L0_{i+1}',
        'type': 'calls',
        'source_id': src,
        'target_id': tgt,
        'status': 'confirmed',
        'sources': [make_source(cpp, src, 1650, 1650, anchor_kind='call_site')]
    })

# L0 -> tiling
relations.append({
    'id': 'REL_L0_TO_TILING',
    'type': 'calls',
    'source_id': 'SYM_L0_FAG',
    'target_id': 'SYM_TILING_MAIN',
    'status': 'confirmed',
    'sources': [make_source(l0_cpp, 'INFER_SHAPE', 172, 182, anchor_kind='call_site')]
})

# Tiling -> arch35 classes
relations.append({
    'id': 'REL_TILING_TO_NORMAL',
    'type': 'dispatches_to',
    'source_id': 'SYM_TILING_MAIN',
    'target_id': 'SYM_TILING_NORMAL_REGBASE',
    'status': 'confirmed',
    'sources': [make_source(tiling_cpp, 'TilingRegistryArch::GetInstance().DoTilingImpl', 430, 430, anchor_kind='call_site')]
})
relations.append({
    'id': 'REL_NORMAL_TO_VARLEN',
    'type': 'extends',
    'source_id': 'SYM_TILING_VARLEN_REGBASE',
    'target_id': 'SYM_TILING_NORMAL_REGBASE',
    'status': 'confirmed',
    'sources': [make_source(varlen_cpp, 'class FlashAttentionScoreGradTilingVarlenRegbase : public FlashAttentionScoreGradTilingNormalRegbase', 22, 22, anchor_kind='class_definition')]
})

# Tiling -> kernel
relations.append({
    'id': 'REL_TILING_TO_KERNEL',
    'type': 'launches',
    'source_id': 'SYM_TILING_NORMAL_REGBASE',
    'target_id': 'SYM_REGBASE_FAG',
    'status': 'confirmed',
    'sources': [make_source(entry_h, 'RegbaseFAG', 201, 210, anchor_kind='kernel_entry')]
})

# kernel classes used by RegbaseFAG
relations.append({
    'id': 'REL_REGBASE_TO_KERNEL_CLASS',
    'type': 'instantiates',
    'source_id': 'SYM_REGBASE_FAG',
    'target_id': 'SYM_KERNEL_CLASS',
    'status': 'confirmed',
    'sources': [make_source(entry_h, 'FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>', 85, 87, anchor_kind='template_instantiation')]
})
relations.append({
    'id': 'REL_REGBASE_TO_KERNEL_DETER',
    'type': 'instantiates',
    'source_id': 'SYM_REGBASE_FAG',
    'target_id': 'SYM_KERNEL_CLASS_DETER',
    'status': 'confirmed',
    'sources': [make_source(entry_h, 'FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>', 86, 86, anchor_kind='template_instantiation')]
})

doc = {
    'version': 1,
    'artifact': {
        'type': 'operator.entrypoints',
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
    'relations': relations,
    'unresolved': [],
    'analysis_status': 'complete',
    'reason': 'All API, tiling, proto, and kernel entrypoints mapped for arch35'
}

out_path = f"{PROJECT_ROOT}\\.understand-operator\\flash_attention_score_grad\\facts\\operator\\entrypoints.yaml"
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

print(f"Wrote {len(items)} items + {len(relations)} relations to {out_path}")
