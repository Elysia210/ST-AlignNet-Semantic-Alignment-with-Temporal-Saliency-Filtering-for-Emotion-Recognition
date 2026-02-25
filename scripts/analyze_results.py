#!/usr/bin/env python3
# scripts/analyze_results.py
"""
分析和汇总实验结果
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path


def load_results(results_dir):
    """加载所有实验结果"""
    results = []
    
    for root, dirs, files in os.walk(results_dir):
        if 'results.json' in files:
            result_file = os.path.join(root, 'results.json')
            
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # 解析实验配置
                parts = Path(root).parts
                
                # 确定数据集和评估模式
                if 'deap_si' in root:
                    dataset = 'DEAP'
                    eval_mode = 'Subject-Independent'
                elif 'deap_sd' in root:
                    dataset = 'DEAP'
                    eval_mode = 'Subject-Dependent'
                elif 'mahnob_si' in root:
                    dataset = 'MAHNOB'
                    eval_mode = 'Subject-Independent'
                elif 'mahnob_sd' in root:
                    dataset = 'MAHNOB'
                    eval_mode = 'Subject-Dependent'
                else:
                    continue
                
                # 提取模态和encoder
                folder_name = os.path.basename(root)
                if folder_name.startswith('eeg_'):
                    modality = 'EEG'
                    encoder = folder_name.replace('eeg_', '')
                elif folder_name.startswith('facial_'):
                    modality = 'Facial'
                    encoder = folder_name.replace('facial_', '')
                else:
                    continue
                
                # 提取结果
                avg_results = data.get('average', {})
                std_results = data.get('std', {})
                
                result_entry = {
                    'Dataset': dataset,
                    'Eval_Mode': eval_mode,
                    'Modality': modality,
                    'Encoder': encoder.upper(),
                    'Valence_Acc': avg_results.get('valence_acc', 0),
                    'Valence_Acc_Std': std_results.get('valence_acc', 0),
                    'Arousal_Acc': avg_results.get('arousal_acc', 0),
                    'Arousal_Acc_Std': std_results.get('arousal_acc', 0),
                    'Avg_Acc': avg_results.get('avg_acc', 0),
                    'Avg_Acc_Std': std_results.get('avg_acc', 0),
                    'Valence_F1': avg_results.get('valence_f1', 0),
                    'Arousal_F1': avg_results.get('arousal_f1', 0),
                    'Avg_F1': avg_results.get('avg_f1', 0),
                    'Valence_MAE': avg_results.get('valence_mae', 0),
                    'Arousal_MAE': avg_results.get('arousal_mae', 0),
                    'Avg_MAE': avg_results.get('avg_mae', 0),
                    'Avg_MAE_Std': std_results.get('avg_mae', 0)
                }
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                continue
    
    return results


def create_summary_table(results, output_file):
    """创建汇总表格"""
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # 排序
    df = df.sort_values(['Dataset', 'Eval_Mode', 'Modality', 'Encoder'])
    
    # 保存CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Summary saved to: {output_file}")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for dataset in df['Dataset'].unique():
        print(f"\n{dataset} Dataset:")
        df_dataset = df[df['Dataset'] == dataset]
        
        for eval_mode in df_dataset['Eval_Mode'].unique():
            print(f"\n  {eval_mode}:")
            df_eval = df_dataset[df_dataset['Eval_Mode'] == eval_mode]
            
            for modality in df_eval['Modality'].unique():
                df_mod = df_eval[df_eval['Modality'] == modality]
                print(f"\n    {modality} Modality:")
                print(f"    {'Encoder':<15} {'Val Acc':<10} {'Aro Acc':<10} {'Avg MAE':<10}")
                print(f"    {'-'*45}")
                
                for _, row in df_mod.iterrows():
                    print(f"    {row['Encoder']:<15} "
                          f"{row['Valence_Acc']:<10.4f} "
                          f"{row['Arousal_Acc']:<10.4f} "
                          f"{row['Avg_MAE']:<10.4f}")
    
    print("\n" + "="*80)


def create_latex_table(results, output_file):
    """生成LaTeX表格"""
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return
    
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{llcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Modality & Encoder & \\multicolumn{3}{c}{\\textbf{DEAP}} & \\multicolumn{3}{c}{\\textbf{MAHNOB}} \\\\")
    latex_lines.append("& & Arousal Acc & Valence Acc & Avg MAE & Arousal Acc & Valence Acc & Avg MAE \\\\")
    latex_lines.append("\\midrule")
    
    for modality in ['EEG', 'Facial']:
        df_mod = df[(df['Modality'] == modality) & (df['Eval_Mode'] == 'Subject-Independent')]
        
        for encoder in df_mod['Encoder'].unique():
            df_encoder = df_mod[df_mod['Encoder'] == encoder]
            
            deap_row = df_encoder[df_encoder['Dataset'] == 'DEAP']
            mahnob_row = df_encoder[df_encoder['Dataset'] == 'MAHNOB']
            
            if len(deap_row) > 0 and len(mahnob_row) > 0:
                deap_row = deap_row.iloc[0]
                mahnob_row = mahnob_row.iloc[0]
                
                latex_lines.append(
                    f"{modality if encoder == df_mod['Encoder'].iloc[0] else ''} & "
                    f"{encoder} & "
                    f"{deap_row['Arousal_Acc']:.3f} & "
                    f"{deap_row['Valence_Acc']:.3f} & "
                    f"{deap_row['Avg_MAE']:.2f} & "
                    f"{mahnob_row['Arousal_Acc']:.3f} & "
                    f"{mahnob_row['Valence_Acc']:.3f} & "
                    f"{mahnob_row['Avg_MAE']:.2f} \\\\"
                )
        
        if modality == 'EEG':
            latex_lines.append("\\midrule")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Experiment 1 Results: Single-Modality Performance (Subject-Independent)}")
    latex_lines.append("\\label{tab:exp1_si}")
    latex_lines.append("\\end{table}")
    
    latex_file = output_file.replace('.csv', '.tex')
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table saved to: {latex_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--latex', action='store_true',
                       help='Also generate LaTeX table')
    
    args = parser.parse_args()
    
    # 加载结果
    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"Found {len(results)} experiment results")
    
    # 创建汇总表格
    create_summary_table(results, args.output_file)
    
    # 生成LaTeX表格
    if args.latex:
        create_latex_table(results, args.output_file)
