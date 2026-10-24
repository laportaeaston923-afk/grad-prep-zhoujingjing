"""Generate a transparent report and standalone plots from measured results."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run import ROOT

MAIN={'WiGesture':'wigesture_paper_v2','WiFall':'wifall_paper_v2','WiCount':'wicount_paper_v2'}
ABLATIONS=['wigesture_paper_v2','wigesture_no_gan','wigesture_no_arl','wigesture_no_time','wigesture_scratch']

def load(name,file='results.json'): return json.loads((ROOT/'runs'/name/file).read_text())

def main():
    charts=ROOT/'reports'; charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    results={d:load(n) for d,n in MAIN.items()}
    extras={d:load(n,'extended_metrics.json') for d,n in MAIN.items()}
    baselines=json.loads((ROOT/'runs/recurrent_baselines.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for col,(dataset,name) in enumerate(MAIN.items()):
        history=[json.loads(x) for x in (ROOT/'runs'/name/'history.jsonl').read_text().splitlines()]
        for row,stage in enumerate(['pretrain','prediction']):
            subset=[x for x in history if x['stage']==stage]
            axes[row,col].plot([x['epoch'] for x in subset],[x['validation']['mse'] for x in subset],color='#147d92',lw=2)
            axes[row,col].set(title=f'{dataset}: {stage}',xlabel='Epoch',ylabel='Validation MSE'); axes[row,col].grid(alpha=.2)
    fig.savefig(charts/'training_curves.png',dpi=170); plt.close(fig)
    methods=['CSI-BERT2','Linear_boundary_hold','IDW_power2','RNN','LSTM','GRU']
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    positions=np.arange(3); width=.13
    for i,method in enumerate(methods):
        values=[]
        for dataset in MAIN:
            if method in ['RNN','LSTM','GRU']: v=baselines[f'{dataset.lower()}_{method.lower()}']['mse']
            else: v=extras[dataset]['regression']['prediction_0.2'][method]['mse']
            values.append(v)
        ax.bar(positions+(i-2.5)*width,values,width,label=method)
    ax.set(xticks=positions,xticklabels=list(MAIN),ylabel='Test MSE',title='Forecasting the final 20%: shared test windows'); ax.legend(ncol=3,fontsize=9); ax.grid(axis='y',alpha=.2)
    fig.savefig(charts/'forecast_comparison.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,(dataset,name) in zip(axes.flat,MAIN.items()):
        for task,rows in extras[dataset]['classification'].items():
            ratios=[float(x) for x in rows]; scores=[100*x['accuracy'] for x in rows.values()]
            ax.plot(ratios,scores,marker='o',label=task)
        ax.set(title=dataset,xlabel='Artificial packet loss',ylabel='Accuracy (%)',ylim=(0,100)); ax.legend(); ax.grid(alpha=.2)
    for ax,(dataset,name) in zip(axes.flat[3:],MAIN.items()):
        for method in ['CSI-BERT2','Linear_boundary_hold','IDW_power2']:
            ratios=[.15,.3,.5,.7]; values=[extras[dataset]['regression'][f'recovery_{r}'][method]['mse'] for r in ratios]
            ax.plot(ratios,values,marker='o',label=method)
        ax.set(title=f'{dataset}: recovery',xlabel='Artificial packet loss',ylabel='MSE'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(charts/'packet_loss_robustness.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    names=ABLATIONS[:-1]; labels=['Full','No GAN','No ARL','No time']
    for ax,stage in zip(axes,['pretrain','prediction']):
        ax.bar(labels,[load(n)[stage]['mse'] for n in names],color=['#147d92','#7fb7be','#dda15e','#a06a80'])
        ax.set(title=f'WiGesture: {stage}',ylabel='Test MSE'); ax.grid(axis='y',alpha=.2)
    fig.savefig(charts/'ablations.png',dpi=170); plt.close(fig)
    lines=['# 继续复现实验报告','',
        '本次在原项目中补充 WiFall、WiCount、论文比例的 WiGesture 重跑、四项消融/对照、循环网络预测基线和丢包鲁棒性评估。所有下表数字均来自本机实际运行，原始权重、配置、每轮日志和 JSON 结果均在 runs/。','',
        '## 三个真实数据集的主要结果','',
        '| 数据集 | 恢复 MSE（随机 15%） | 预测 MSE（末尾 20%） | 预测 SMAPE | 预测 MAPE |','|---|---:|---:|---:|---:|']
    for d,r in results.items(): lines.append(f"| {d} | {r['pretrain']['mse']:.4f} | {r['prediction']['mse']:.4f} | {r['prediction']['smape']:.4f} | {r['prediction']['mape']:.4f} |")
    lines+=['','| 数据集 | 任务 | 100Hz 准确率 | 宏平均 F1 | 100Hz 训练→50Hz 测试 | 测试样本数 |','|---|---|---:|---:|---:|---:|']
    tasknames={'action':'动作/手势识别','people':'人员识别','fall':'跌倒检测'}
    for d,r in results.items():
        for task,values in r.items():
            if task not in tasknames: continue
            label='人数估计' if d=='WiCount' else tasknames[task]
            gap=values['50hz_test'].get('accuracy'); gap=f'{gap:.2%}' if gap is not None else '无可用窗口'
            cm=np.asarray(values['confusion_matrix'],dtype=float); tp=cm.diagonal(); precision=tp/np.maximum(cm.sum(0),1); recall=tp/np.maximum(cm.sum(1),1); f1=np.mean(2*precision*recall/np.maximum(precision+recall,1e-8))
            lines.append(f"| {d} | {label} | {values['accuracy']:.2%} | {f1:.4f} | {gap} | {values['samples']} |")
    fall=results['WiFall']['fall']
    lines+=['',f"WiFall 跌倒类（类别 0）的精确率为 {fall['fall_precision']:.2%}，召回率为 {fall['fall_recall']:.2%}，二分类平衡准确率为 {fall['balanced_accuracy']:.2%}。这些指标用于区分模型是否仅受类别比例影响。",'',
        '公开 WiFall 压缩包的人员分布极不均衡：ID2 有 640 个录制文件，其余每个 ID 各 32 个。因此人员识别总体准确率会受多数类明显影响，需结合宏平均 F1 和混淆矩阵评估。项目保留原始分布，没有把多数类重命名或伪装成均衡数据。']
    lines+=['','## 在相同测试窗口上比较预测方法','',
        '| 数据集 | CSI-BERT2 | Linear（边界保持） | IDW | RNN | LSTM | GRU |','|---|---:|---:|---:|---:|---:|---:|']
    for d in MAIN:
        rows=extras[d]['regression']['prediction_0.2']; values=[rows[k]['mse'] for k in methods[:3]]+[baselines[f'{d.lower()}_{k.lower()}']['mse'] for k in methods[3:]]
        lines.append('| '+d+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','线性插值在末尾预测时退化为最后一个观测值保持；IDW 使用所有已观测时间点。神经网络、插值和 IDW 只读取可观测上下文，不读取被遮挡真值。循环网络为独立明确配置：单层 hidden=128，读取前 80 点，线性头一次输出后 20 点，AdamW 0.0005、30 轮，以验证集 MSE 选权重。它们不是作者发布的基线实现，参数量也不与论文表格逐项相同，因此用于本机可重复对照，不能宣称复现了论文全部基线配置。','',
        '## WiGesture 消融','',
        '| 设置 | 恢复 MSE | 预测 MSE（20%） | 手势准确率 |','|---|---:|---:|---:|']
    for n,label in zip(ABLATIONS,['完整模型','关闭 GAN','关闭 ARL','关闭时间嵌入','无预训练，从头训练分类']):
        r=load(n); rec=f"{r['pretrain']['mse']:.4f}" if 'pretrain' in r else '—'; pred=f"{r['prediction']['mse']:.4f}" if 'prediction' in r else '—'
        lines.append(f"| {label} | {rec} | {pred} | {r['action']['accuracy']:.2%} |")
    lines+=['','消融每项独立训练 30 轮、随机种子 42，关闭的模块在训练和推理时均关闭。这里只提供单种子的观察，不据此断言消融差异有统计显著性。无预训练对照从头训练整个分类模型，而预训练模型的分类阶段冻结主干；这个对照同时改变初始化和可训练参数范围，不能把差异完全归因于预训练。','',
        '## 与论文数字的距离','',
        '| 数据集 | 论文恢复 MSE | 本机恢复 MSE | 论文预测 MSE | 本机预测 MSE |','|---|---:|---:|---:|---:|']
    references={'WiGesture':(2.0800,3.2942),'WiFall':(4.1463,4.8598),'WiCount':(2.4531,5.3401)}
    for d,(rec,pred) in references.items(): lines.append(f"| {d} | {rec:.4f} | {results[d]['pretrain']['mse']:.4f} | {pred:.4f} | {results[d]['prediction']['mse']:.4f} |")
    lines+=['','论文数字来自附件 Table IV / VI，DOI: [10.1109/TMC.2025.3640420](https://doi.org/10.1109/TMC.2025.3640420)。此表供定位差距；由于切分、训练窗口和预算不同，不是严格同协议排名。','',
        '## 本轮修正及协议','',
        '- 论文 MLM 掩码为 15%–70%，预测训练末尾掩码为 15%–40%，预测测试为末尾 20%。新增 `--paper-mask` 对齐该设置；第一轮历史结果 wig_paper_config 使用 10%–70% 的 MLM 和 15% 的预测测试，继续保留。','- WiGesture、WiCount 每个录制按时间划分为 81%/9%/10%；WiFall 单文件只有约数秒，因此按每个人/动作组内的录制文件顺序划分为 81%/9%/10%，同一录制文件不会跨集合。论文为 90%/10%，没有单独验证集；本项目留出验证集，所有模型都用相同协议。','- 训练按不重叠 100 点窗口，每轮至多 2000 个；保留原始真实丢包并排除无效目标。50Hz 使用间隔 2 的 100 点窗口，覆盖约两秒。没有实现 Table VII 的 100Hz+50Hz 混合训练设置。','- 丢包率评估覆盖恢复 15%/30%/50%/70%、预测 15%/20%/30%/40%、分类额外丢包 0%/15%/30%/50%/70%。`comparison.csv` 保存全部逐项数字，`extended_metrics.json` 保存混淆矩阵。','- 公开 WiGesture 包缺 ID5/waveright.csv。WiCount 共 15 份、WiFall 共 928 份录制，原始 zip 和 SHA256 记录在数据目录。','',
        '## 作者公开权重核对','',
        '另保存了 https://huggingface.co/RS2002/CSI-BERT2 的公开权重到 author_checkpoint/。严格匹配本项目主干的 133 个张量，并在真实数据上验证 1×100×128 特征输出为有限值；未执行下载的 model.py。该文件只有特征主干，没有恢复、预测或分类任务头，不能直接拿来计算论文任务误差。公开权重的精确训练/测试划分也未确认，因此没有混入本机独立测试指标。详情见 author_checkpoint/inspection.json。','',
        '## 尚未完成','',
        'CommPre 的原始 892 个仿真场景数据、场景文件及生成配置没有在当前可访问的作者公开仓库/数据集列表中找到，因此没有用替代随机数据冒充该数据集。仍未完成：CommPre、全部论文基线（例如 Mamba/OCEAN/CV-3DCNN/ConvLSTM）、下游模型在恢复数据上的训练矩阵、混合采样率训练、多随机种子统计，以及完全匹配论文训练集和收敛策略的数值复现。','',
        '本轮已完成三个公开真实数据集上的核心训练、评估和选定基线/消融；这仍不等于论文所有表格与指标均已复现。','',
        '## 复查和重跑','',
        '在 E:\\all\\CSI-BERT2-reproduction 中执行 `continue_reproduction.ps1` 可检查已完成实验并运行缺失项；已完成的训练会跳过。若要新的实验参数或随机种子，请使用新的 run-name。','',
        '单项示例：`.\\.venv\\Scripts\\python.exe run.py --dataset WiFall --paper-mask --stage all --epochs 30 --run-name wifall_new`。','',
        '推理示例：`.\\.venv\\Scripts\\python.exe infer.py --input runs/wifall_paper_v2/inference_examples/input.npz --output fall_result.npz --task fall --run-name wifall_paper_v2`。','',
        '## 图表','',
        '![训练曲线](reports/training_curves.png)','',
        '![预测方法比较](reports/forecast_comparison.png)','',
        '![丢包鲁棒性](reports/packet_loss_robustness.png)','',
        '![消融](reports/ablations.png)']
    (ROOT/'继续复现报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(ROOT.glob('*.py'))+list((ROOT/'src').glob('*.py'))}
    (ROOT/'reproduction_code_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(ROOT/'继续复现报告.md')

if __name__=='__main__': main()
