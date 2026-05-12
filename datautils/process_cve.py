"""
IDA script for CVE artifact feature extraction (no binaryai dependency).
Run via: idat64.exe -c -A -S"process_cve.py" -o"out.idb" "binary"
Output pkl format: {func_name: [func_addr, asm_list, rawbytes_list, cfg, None]}
"""
import idc
import idautils
import idaapi
import pickle
import networkx as nx
import os
import sys

# injected by extract_cve.py via -S"script.py KEY=value"
# fallback: derive from input file path
def get_save_path():
    # IDA passes script args after the script name separated by space
    # e.g. -S"process_cve.py SAVE_PATH=/some/path.pkl"
    for arg in idc.ARGV:
        if arg.startswith('SAVE_PATH='):
            return arg[len('SAVE_PATH='):]
    # fallback: save next to input binary
    binary_path = idc.get_input_file_path()
    return binary_path + '_extract.pkl'


def get_cfg(func):
    func_addr_set = set(idautils.FuncItems(func))
    nx_graph = nx.DiGraph()
    flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
    for block in flowchart:
        curr_addr = block.start_ea
        if curr_addr not in func_addr_set:
            continue
        asm, raw = [], b""
        while curr_addr < block.end_ea:
            asm.append(idc.GetDisasm(curr_addr))
            sz = idc.get_item_size(curr_addr)
            raw += idc.get_bytes(curr_addr, sz) or b'\x00' * sz
            curr_addr = idc.next_head(curr_addr, block.end_ea)
        nx_graph.add_node(block.start_ea, asm=asm, raw=raw)
        for pred in block.preds():
            if pred.start_ea in func_addr_set:
                nx_graph.add_edge(pred.start_ea, block.start_ea)
        for succ in block.succs():
            if succ.start_ea in func_addr_set:
                nx_graph.add_edge(block.start_ea, succ.start_ea)
    return nx_graph


def extract_all():
    saved = {}
    for func in idautils.Functions():
        seg_name = idc.get_segm_name(func)
        if seg_name in ['.plt', 'extern', '.init', '.fini']:
            continue
        func_name = idc.get_func_name(func)
        if not func_name:
            continue
        idc.create_insn(func)
        idc.add_func(func)

        asm_list = [idc.GetDisasm(i) for i in idautils.FuncItems(func)]
        raw = b"".join(
            idc.get_bytes(i, idc.get_item_size(i)) or b''
            for i in idautils.FuncItems(func)
        )
        cfg = get_cfg(func)
        saved[func_name] = [func, asm_list, raw, cfg, None]
    return saved


if __name__ == '__main__':
    idc.auto_wait()
    save_path = get_save_path()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = extract_all()
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'[+] saved {len(data)} functions -> {save_path}')
    idc.qexit(0)
