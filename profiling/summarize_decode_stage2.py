"""Aggregate all samples without silently choosing a favorable run."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    groups = {
        'qwen_rms':['rms_before','rms_after','rms_hybrid_control','rms_final_control'],
        'rms_int4_head':['rms_int4_head','rms_int4_head_b'],
        'rms_int8':['rms_int8','rms_int8_b'],
        'rms_int8_int4_head':['rms_int8_int4_head'],
        'hybrid':['hybrid_a','hybrid_b'],
    }
    result = {'method':'pooled median of all listed unprofiled CUDA-event samples; each decode sample averages 16 tokens; uneven run counts, no confidence intervals', 'groups':{}}
    for group,names in groups.items():
        reports = [json.loads((args.root/(name+'.json')).read_text()) for name in names]
        rows = []
        for phase in ('prefill','decode'):
            for length in (2048,4096):
                measurements = [r for report in reports for r in report['measurements'] if r['phase']==phase and r['initial_context']==length]
                samples = [x for r in measurements for x in r['gpu_samples_ms']]
                rows.append({'phase':phase,'context':length,'samples':len(samples),
                    'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples),
                    'run_medians_ms':[r['gpu_median_ms'] for r in measurements]})
        result['groups'][group] = {'files':names,'measurements':rows}
    base = result['groups']['qwen_rms']['measurements']
    for group,data in result['groups'].items():
        for row,reference in zip(data['measurements'],base):
            row['latency_reduction_vs_rms_pct'] = 100*(1-row['median_ms']/reference['median_ms'])
            row['inverse_latency_increase_vs_rms_pct'] = 100*(reference['median_ms']/row['median_ms']-1)
    (args.root/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,data in result['groups'].items():
        print(name,[(r['phase'],r['context'],round(r['median_ms'],3),round(r['latency_reduction_vs_rms_pct'],2)) for r in data['measurements']])


if __name__=='__main__':
    main()
