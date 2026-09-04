"""Pool every unprofiled sample; never select the fastest repeat."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args()
    samples=defaultdict(list); files=defaultdict(set)
    for path in args.root.glob('benchmark*.json'):
        report=json.loads(path.read_text()); group='hybrid' if report.get('hybrid') else 'rms'
        for row in report['measurements']:
            key=(group,row['context'],row['variant'])
            samples[key].extend(row['gpu_samples_ms_per_token']); files[key].add(path.name)
    result={'method':'pooled median of every unprofiled CUDA-event sample found in benchmark*.json; each sample averages 16 sequential tokens; no confidence intervals','measurements':[]}
    for key,values in sorted(samples.items()):
        group,context,variant=key
        dynamic=statistics.median(samples[(group,context,'dynamic')])
        median=statistics.median(values)
        result['measurements'].append({'group':group,'context':context,'variant':variant,
            'samples':len(values),'median_ms_per_token':median,'min_ms':min(values),'max_ms':max(values),
            'latency_reduction_vs_dynamic_pct':100*(1-median/dynamic),
            'serial_tokens_per_second':1000/median,'files':sorted(files[key])})
    baselines=[]
    for path in sorted(args.root.glob('plain_baseline_*.json')):
        report=json.loads(path.read_text())
        baselines.extend(x for row in report['measurements'] if row['phase']=='decode' and row['initial_context']==2048 for x in row['gpu_samples_ms'])
    if baselines:
        plain=statistics.median(baselines)
        fused=statistics.median(samples[('hybrid',2048,'fused')])
        result['plain_baseline_2048']={'samples':len(baselines),'median_ms_per_token':plain,
            'hybrid_fused_ms_per_token':fused,'latency_reduction_pct':100*(1-fused/plain),
            'serial_throughput_increase_pct':100*(plain/fused-1),
            'warning':'different harnesses/processes; dynamic-vs-fused comparisons within each benchmark are stronger evidence'}
    (args.root/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
