import re
import json
import csv

log_text = r'''
🚀 [TRAINING SUBPROCESS] DORA r=4 seed=42
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r4_seed42_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=4 seed=42
✅ [EVALUATION COMPLETE] Acc: 40.00% | Time: 907.0s

🚀 [TRAINING SUBPROCESS] LORA r=8 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r8_seed123_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=8 seed=123
✅ [EVALUATION COMPLETE] Acc: 50.00% | Time: 710.5s

🚀 [TRAINING SUBPROCESS] DORA r=8 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r8_seed123_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=8 seed=123
✅ [EVALUATION COMPLETE] Acc: 50.00% | Time: 909.7s

🚀 [TRAINING SUBPROCESS] LORA r=4 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r4_seed123_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=4 seed=123
✅ [EVALUATION COMPLETE] Acc: 55.00% | Time: 741.9s

🚀 [TRAINING SUBPROCESS] LORA r=4 seed=456
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r4_seed456_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=4 seed=456
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 715.4s

🚀 [TRAINING SUBPROCESS] LORA r=8 seed=456
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r8_seed456_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=8 seed=456
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 702.4s

🚀 [TRAINING SUBPROCESS] LORA r=16 seed=42
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r16_seed42_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=16 seed=42
✅ [EVALUATION COMPLETE] Acc: 40.00% | Time: 693.9s

🚀 [TRAINING SUBPROCESS] LORA r=16 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r16_seed123_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=16 seed=123
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 721.6s

🚀 [TRAINING SUBPROCESS] LORA r=16 seed=456
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r16_seed456_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=16 seed=456
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 716.0s

🚀 [TRAINING SUBPROCESS] LORA r=32 seed=42
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r32_seed42_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=32 seed=42
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 705.7s

🚀 [TRAINING SUBPROCESS] LORA r=32 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r32_seed123_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=32 seed=123
✅ [EVALUATION COMPLETE] Acc: 50.00% | Time: 722.8s

🚀 [TRAINING SUBPROCESS] LORA r=32 seed=456
✅ [TRAINING COMPLETE] Saved to /kaggle/working/lora_r32_seed456_steps20
🚀 [EVALUATION SUBPROCESS] LORA r=32 seed=456

🚀 [TRAINING SUBPROCESS] DORA r=16 seed=42
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r16_seed42_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=16 seed=42
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 856.7s

🚀 [TRAINING SUBPROCESS] DORA r=16 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r16_seed123_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=16 seed=123
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 913.5s

🚀 [TRAINING SUBPROCESS] DORA r=16 seed=456
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r16_seed456_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=16 seed=456
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 847.5s

🚀 [TRAINING SUBPROCESS] DORA r=32 seed=42
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r32_seed42_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=32 seed=42
✅ [EVALUATION COMPLETE] Acc: 45.00% | Time: 911.7s

🚀 [TRAINING SUBPROCESS] DORA r=32 seed=123
✅ [TRAINING COMPLETE] Saved to /kaggle/working/dora_r32_seed123_steps20
🚀 [EVALUATION SUBPROCESS] DORA r=32 seed=123

🚀 [EVALUATION SUBPROCESS] LORA r=8 seed=42
✅ [EVALUATION COMPLETE] Acc: 50.00% | Time: 123.0s

🚀 [EVALUATION SUBPROCESS] DORA r=8 seed=42
✅ [EVALUATION COMPLETE] Acc: 40.00% | Time: 123.0s
'''

def reconstruct_from_logs(text):
    results = {}
    current_key = None
    for line in text.split('\n'):
        m_start = re.search(r'\[(?:TRAINING|EVALUATION) SUBPROCESS\]\s+([A-Za-z]+)\s+r=(\d+)\s+seed=(\d+)', line)
        if m_start:
            method, rank, seed = m_start.group(1).lower(), int(m_start.group(2)), int(m_start.group(3))
            current_key = (method, rank, seed)
            if current_key not in results:
                results[current_key] = {'accuracy': None}
                
        m_eval = re.search(r'\[EVALUATION COMPLETE\] Acc:\s*([\d\.]+)%', line)
        if m_eval and current_key:
            results[current_key]['accuracy'] = float(m_eval.group(1)) / 100.0
    return results

file_results = reconstruct_from_logs(log_text)

METHODS = ['lora', 'dora']
RANKS = [4, 8, 16, 32]
SEEDS = [42, 123, 456]

csv_data = []
json_data = []

print('='*120)
print(f'{"Method":<8} | {"Rank":<4} | {"Seed":<4} | {"Logged Acc":<10} | {"Artifact?":<10} | {"Rerun Req?":<10} | {"Confidence":<20}')
print('-' * 120)

recovered_count = 0
rerun_count = 0

for m in METHODS:
    for r in RANKS:
        for s in SEEDS:
            key = (m, r, s)
            recov = file_results.get(key)
            
            artifact_available = False
            
            if recov and recov['accuracy'] is not None:
                recovered_count += 1
                logged_accuracy = recov['accuracy']
                rerun_required = False
                result_confidence = 'VERIFIED_FROM_LOG'
                logged_result_source = 'parsed_from_text_logs'
                acc_str = f'{logged_accuracy*100:.2f}%'
            else:
                logged_accuracy = None
                rerun_required = True
                result_confidence = 'NONE'
                logged_result_source = None
                acc_str = 'N/A'
                rerun_count += 1
                
            entry = {
                'method': m,
                'rank': r,
                'seed': s,
                'logged_accuracy': logged_accuracy,
                'logged_result_source': logged_result_source,
                'artifact_available': artifact_available,
                'rerun_required': rerun_required,
                'result_confidence': result_confidence
            }
            
            json_data.append(entry)
            csv_data.append([
                m, r, s, logged_accuracy, logged_result_source, 
                artifact_available, rerun_required, result_confidence
            ])
            
            print(f'{m.upper():<8} | {r:<4} | {s:<4} | {acc_str:<10} | {str(artifact_available):<10} | {str(rerun_required):<10} | {result_confidence:<20}')

print('-' * 120)
print(f'Results recovered from logs (No Rerun Needed): {recovered_count}/24')
print(f'Runs genuinely missing (Rerun Required):         {rerun_count}/24')
print('='*120)

with open('pilot_recovery_manifest.json', 'w') as f:
    json.dump(json_data, f, indent=2)
    
with open('pilot_recovery_manifest.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'rank', 'seed', 'logged_accuracy', 'logged_result_source', 'artifact_available', 'rerun_required', 'result_confidence'])
    writer.writerows(csv_data)
