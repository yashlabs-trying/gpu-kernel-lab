"""Summarize Nsight SQLite data without pretending launch metadata is counters."""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def category(name):
    lower = name.lower()
    if 'flash' in lower:
        return 'flash_attention'
    if 'gemm' in lower or 'gemv' in lower:
        return 'gemm_gemv'
    if 'catarray' in lower:
        return 'concatenation'
    if 'reduce' in lower or 'reduction' in lower:
        return 'reduction'
    if 'copy' in lower:
        return 'copy_or_cast'
    return 'other_elementwise_or_misc'


def summarize(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    kernels = list(db.execute('SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start'))
    union = 0
    left, right = kernels[0]['start'], kernels[0]['end']
    by_name, categories = {}, defaultdict(float)
    for kernel in kernels:
        if kernel['start'] > right:
            union += right - left
            left, right = kernel['start'], kernel['end']
        else:
            right = max(right, kernel['end'])
        duration = kernel['end'] - kernel['start']
        name = kernel['name']
        categories[category(name)] += duration / 1e6
        if name not in by_name:
            by_name[name] = {'name': name, 'count': 0, 'total_ms': 0,
                             'registers_per_thread': set(), 'shared_bytes': set(),
                             'block_threads': set(), 'local_bytes_per_thread': set()}
        entry = by_name[name]
        entry['count'] += 1
        entry['total_ms'] += duration / 1e6
        entry['registers_per_thread'].add(kernel['registersPerThread'])
        entry['shared_bytes'].add(kernel['staticSharedMemory'] + kernel['dynamicSharedMemory'])
        entry['block_threads'].add(kernel['blockX'] * kernel['blockY'] * kernel['blockZ'])
        entry['local_bytes_per_thread'].add(kernel['localMemoryPerThread'])
    union += right - left
    span = max(k['end'] for k in kernels) - min(k['start'] for k in kernels)
    total = sum(k['end'] - k['start'] for k in kernels)
    for entry in by_name.values():
        for field, value in list(entry.items()):
            if isinstance(value, set):
                entry[field] = sorted(value)
        entry['mean_us'] = entry['total_ms'] * 1000 / entry['count']
    return {'trace': path.name, 'kernel_count': len(kernels),
            'kernel_time_sum_ms': total / 1e6, 'kernel_busy_union_ms': union / 1e6,
            'first_to_last_kernel_span_ms': span / 1e6,
            'kernel_busy_fraction_of_span': union / span,
            'categories_ms': dict(categories),
            'kernels': sorted(by_name.values(), key=lambda e: -e['total_ms']),
            'limitations': 'Busy fraction is a timeline measure, NOT achieved SM occupancy. Local memory metadata is not measured spill traffic. Profiler overhead changes timings.'}


def main():
    root = Path('results/profiling_20260831')
    results = [summarize(path) for path in sorted(root.glob('*.sqlite'))]
    (root / 'timeline_summary.json').write_text(json.dumps(results, indent=2) + '\n')
    for entry in results:
        print(json.dumps({k: v for k, v in entry.items() if k != 'kernels'}))


if __name__ == '__main__':
    main()
