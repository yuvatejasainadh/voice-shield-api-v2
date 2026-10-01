import json
with open('benchmark_results/wavlm_sliding_window.json') as f:
    d = json.load(f)
lat = d['latency']
print('Latency stats:')
for k,v in lat.items():
    print(f'  {k}: {v:.2f}')
print()
print('Per-file comparison table:')
print(f"{'filename':<12} {'GT':<8} {'full_score':<11} {'max_win':<9} {'mean_win':<10} {'top2':<9} {'top25':<9} {'majority':<9} {'full_pred':<11} {'max_pred'}")
for r in d['per_file']:
    print(f"{r['filename']:<12} {r['ground_truth']:<8} {r['full_length_score']:<11.4f} {r['max_window_score']:<9.4f} {r['mean_window_score']:<10.4f} {r['top2_mean_score']:<9.4f} {r['top25_mean_score']:<9.4f} {r['majority_vote']:<9} {r['full_length_prediction']:<11} {r['max_window_prediction']}")
print()
print('Spoof files MISSED by full-length but detected by MAX window:')
for r in d['per_file']:
    if r['ground_truth']=='spoof' and r['full_length_prediction']=='genuine' and r['max_window_score']>0.5:
        print(f"  {r['filename']}: full={r['full_length_score']:.4f}, max_win={r['max_window_score']:.4f}")
print()
print('Spoof files NOT detected even by MAX window (persistent false negatives):')
for r in d['per_file']:
    if r['ground_truth']=='spoof' and r['max_window_prediction']=='genuine':
        print(f"  {r['filename']}: full={r['full_length_score']:.4f}, max_win={r['max_window_score']:.4f}")
print()
print('Genuine files that became FALSE POSITIVES under MAX window:')
for r in d['per_file']:
    if r['ground_truth']=='genuine' and r['max_window_prediction']=='spoof':
        print(f"  {r['filename']}: full={r['full_length_score']:.4f}, max_win={r['max_window_score']:.4f}")
