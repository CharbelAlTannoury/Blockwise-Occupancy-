from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from cwru_analysis import METHODS, record_floor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
PUB = ROOT / 'publication'
FIG = PUB / 'figures'
TAB = PUB / 'tables'


def table(name, columns, rows, alignment):
    text = '\\begin{tabular}{' + alignment + '}\n\\toprule\n'
    text += ' & '.join(columns) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str,row)) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (TAB / f'cwru_{name}.tex').write_text(text)


def save_figure(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'], 'font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,
                         'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
                         'savefig.facecolor':'white'})
    manifest = json.loads((ROOT/'data/data_manifest.json').read_text())
    meta = {x['file_id']:x for x in manifest}
    m = pd.read_csv(OUT/'metrics.csv'); p = m[m.design=='Primary']
    thresholds = pd.read_csv(OUT/'thresholds.csv')
    t = thresholds[thresholds.design=='Primary'].set_index('load_hp')
    blocks = pd.read_csv(OUT/'primary_blocks.csv.gz')
    summaries = p.groupby(['method','cohort']).agg(high_rate=('high_rate','mean'),warning_rate=('warning_rate','mean'),
                                                n_records=('file_id','size'),blocks=('blocks','sum'),high=('high','sum')).reset_index()
    summaries.to_csv(OUT/'cohort_summaries.csv',index=False)
    rates = summaries.pivot(index='method',columns='cohort',values='high_rate')
    rows=[]
    for load in range(4):
        healthy = p[(p.cohort=='normal')&(p.load_hp==load)].set_index('method')
        rows.append([load,int(t.loc[load,'a']),int(t.loc[load,'k_direct']),int(t.loc[load,'k_record']),
                     f"{int(healthy.loc['Direct count','blocks']):,}",
                     f"{100*healthy.loc['Direct count','high_rate']:.3f}",f"{100*healthy.loc['Record count','high_rate']:.3f}"])
    table('healthy',['Load (hp)',r'$a$',r'$k_{\rm dir}$',r'$k_{\rm rec}$','Blocks',r'Direct (\%)',r'Record (\%)'],rows,'rrrrrrr')
    table('comparators',['Rule',r'Healthy (\%)',r'Primary faults (\%)',r'Other OR (\%)'],
          [[name]+[f'{100*rates.loc[name,c]:.3f}' for c in ['normal','primary','additional']] for name in METHODS],'lrrr')
    grid_index = [(fault,size) for fault in ['IR','B','OR'] for size in [.007,.014,.021]]
    fig, axes = plt.subplots(1,2,figsize=(6.45,3.25),layout='constrained',sharey=True)
    for ax,name in zip(axes,['Direct count','Record count']):
        values = p[(p.cohort=='primary')&(p.method==name)].pivot_table(index=['fault_type','diameter_in'],columns='load_hp',values='high_rate').reindex(grid_index)
        im=ax.imshow(values.values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_title(name);ax.set_xticks(range(4),['0','1','2','3']);ax.set_xlabel('Load (hp)')
        ax.set_yticks(range(9),[f'{f}  {s:.3f} in.' for f,s in grid_index])
        for row in range(9):
            for col in range(4):
                val=values.iloc[row,col]
                ax.text(col,row,f'{100*val:.1f}',ha='center',va='center',fontsize=8.2,color='white' if val>.7 else '#182e42')
        ax.axhline(2.5,color='white',lw=1.7);ax.axhline(5.5,color='white',lw=1.7)
        ax.tick_params(axis='both',length=0)
    cb=fig.colorbar(im,ax=axes,shrink=.92,pad=.025)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1));cb.set_label('High-occupancy fraction')
    save_figure(fig,'cwru_fault_grid')
    for cohort in ['primary','additional']:
        rows=[]
        ss=p[(p.cohort==cohort)&(p.method=='Record count')].sort_values(['fault_type','diameter_in','outer_position','load_hp'])
        for _,z in ss.iterrows():
            item=meta[int(z.file_id)];direct=p[(p.file_id==z.file_id)&(p.method=='Direct count')].iloc[0]
            fault={'IR':'IR','B':'Ball','OR':'OR ('+str(item['outer_position'])+' h)'}[item['fault_type']]
            rows.append([int(z.file_id),fault,f'{z.diameter_in:.3f}',int(z.load_hp),f'{int(z.blocks):,}',
                         f'{100*z.warning_rate:.2f}',f'{100*direct.high_rate:.2f}',f'{100*z.high_rate:.2f}'])
        table(f'{cohort}_records',['File','Defect','Size (in.)','hp','Blocks','Warning','Direct','Record'],rows,'rlrrrrrr')
    sens=m[m.method.isin(['Direct count','Record count'])].groupby(['design','method','cohort']).high_rate.mean().unstack()
    sens.reset_index().to_csv(OUT/'sensitivity_summaries.csv',index=False)
    names=['Primary','r5','r20','m40','gap500','K399','Long training','Student score','Spaced evaluation']
    display={'Primary':'Primary', 'r5':r'$r=q=5$', 'r20':r'$r=q=20$', 'm40':r'$m=40$', 'gap500':'Gap 500',
             'K399':r'$K=399$', 'Long training':'Longer training', 'Student score':r'Student $t_5$ score','Spaced evaluation':'Spaced evaluation'}
    rows=[]
    for name in names:
        tt=thresholds[thresholds.design==name]
        rows.append([display[name],int(tt.record_active.sum())]+[f'{100*sens.loc[(name,method),cohort]:.3f}' for cohort in ['normal','primary'] for method in ['Direct count','Record count']])
    table('sensitivity',['Design','Active / 4','Healthy D','Healthy R','Fault D','Fault R'],rows,'lrrrrr')
    gamma=pd.read_csv(OUT/'record_benchmarks.csv'); gm=gamma.groupby(['gamma','cohort']).high_rate.mean().unstack()
    gm.to_csv(OUT/'record_benchmark_summaries.csv')
    table('record_floors',[r'$\gamma$',r'$k^{\rm bench}_{20}$',r'Healthy (\%)',r'Faults (\%)'],
          [[f'{g:.2f}',record_floor(20,gamma=g)[0],f'{100*gm.loc[g,"normal"]:.3f}',f'{100*gm.loc[g,"primary"]:.3f}'] for g in gm.index],'rrrr')
    matched=pd.read_csv(OUT/'common_nominal_rank.csv').groupby(['statistic','cohort']).rate.mean().unstack()
    matched.to_csv(OUT/'common_rank_summaries.csv')
    table('common_rank',['Statistic',r'Healthy (\%)',r'Primary faults (\%)'],
          [[name,f'{100*matched.loc[name,"normal"]:.3f}',f'{100*matched.loc[name,"primary"]:.3f}'] for name in ['Count']+METHODS[2:]],'lrr')
    exclusions=[('All primary recordings',[]),('Omit reported noise / clipping',[177,191,214,215,228,229]),
                ('Omit all listed acquisition issues',[177,189,201,213,226,238,191,214,215,228,229])]
    exrows=[];excsv=[]
    for name,ids in exclusions:
        ss=p[(p.cohort=='primary')&~p.file_id.isin(ids)]
        means=ss.groupby('method').high_rate.mean()
        exrows.append([name,len(ss.file_id.unique()),f'{100*means["Direct count"]:.3f}',f'{100*means["Record count"]:.3f}'])
        excsv.extend([dict(analysis=name,method=method,n_recordings=len(ss.file_id.unique()),high_rate=float(val),omitted_files=' '.join(map(str,ids))) for method,val in means.items()])
    table('quality',['Set','Files',r'Direct (\%)',r'Record (\%)'],exrows,'lrrr')
    pd.DataFrame(excsv).to_csv(OUT/'quality_sensitivity.csv',index=False)
    diag=pd.read_csv(OUT/'reference_diagnostics.csv')
    table('diagnostics',['Load (hp)',r'$c_0$',r'$\widehat p$',r'$\widehat{\mathrm{Var}}(N)$',r'$20\widehat p(1-\widehat p)$',r'Max. $|\widehat\rho_Z|$'],
          [[int(z.load_hp),f'{t.loc[z.load_hp,"c0"]:.4f}',f'{z.calibration_exceedance_rate:.4f}',f'{z.calibration_count_variance:.4f}',
            f'{z.binomial_variance_at_empirical_rate:.4f}',f'{z.max_abs_score_acf_1_20:.4f}'] for _,z in diag.iterrows()],'rrrrrr')
    coeff=pd.read_csv(OUT/'model_coefficients.csv');coeff=coeff[coeff.design=='Primary']
    table('model_fit',['Load',r'$\bar y$',r'$s_y$',r'$\widehat b_0$',r'$\widehat b_1$',r'$\widehat b_2$',r'$\widehat v_0$',r'$\widehat v_1$',r'$\widehat I$'],
          [[int(z.load_hp)]+[f'{z[c]:.5f}' for c in ['center','scale','b0','b1','b2','v0','v1','information']] for _,z in coeff.iterrows()],'rrrrrrrrr')
    fig,axes=plt.subplots(2,2,figsize=(6.45,3.25),sharex=True,sharey=True,layout='constrained')
    distrows=[]
    for load,ax in enumerate(axes.ravel()):
        for cohort,color in [('normal','#516475'),('primary','#246f93')]:
            ids=[x['file_id'] for x in manifest if x['load_hp']==load and x['cohort']==cohort]
            pmfs=[np.bincount(blocks.loc[blocks.file_id==i,'Count'].to_numpy(dtype=int),minlength=21)/len(blocks[blocks.file_id==i]) for i in ids]
            pmf=np.mean(pmfs,axis=0)
            ax.plot(range(21),pmf,marker='o',ms=2.3,lw=1.2,color=color,label='Healthy' if cohort=='normal' else 'Primary faults')
            distrows.extend(dict(load_hp=load,cohort=cohort,count=k,probability=float(v)) for k,v in enumerate(pmf))
        ax.axvline(t.loc[load,'k_direct']-.5,color='#b36a3b',ls=':',lw=1.2,label='Direct cutoff')
        ax.axvline(t.loc[load,'k_record']-.5,color='#383952',ls='--',lw=1.2,label='Record cutoff')
        ax.set_title(f'{load} hp');ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Exceedance count');ax.set_ylabel('Proportion')
    fig.legend(*axes[0,0].get_legend_handles_labels(),frameon=False,fontsize=8,ncol=4,loc='outside upper center')
    save_figure(fig,'cwru_count_distributions')
    pd.DataFrame(distrows).to_csv(OUT/'count_distributions.csv',index=False)
    acfs=pd.read_csv(OUT/'reference_autocorrelations.csv');stability=pd.read_csv(OUT/'within_record_stability.csv')
    fig,axes=plt.subplots(2,2,figsize=(6.45,3.65),layout='constrained')
    colors=['#246f93','#b36a3b','#649073','#8e6c99']
    for load,color in enumerate(colors):
        for ax,series in zip(axes[0],['innovations','micro_scores']):
            z=acfs[(acfs.load_hp==load)&(acfs.series==series)&(acfs.lag>0)]
            ax.plot(z.raw_lag/48,z.acf,lw=.9,color=color,label=f'{load} hp')
        ss=stability[(stability.cohort=='normal')&(stability.load_hp==load)]
        axes[1,0].plot(ss.bin,ss.mean_count,'o-',lw=1,ms=3,color=color)
        axes[1,1].plot(ss.bin,100*ss.record_high_rate,'o-',lw=1,ms=3,color=color)
    for ax,title in zip(axes[0],['Innovation autocorrelation','Micro-score autocorrelation']):
        ax.set_title(title);ax.axhline(0,color='.6',lw=.6);ax.set_xlabel('Lag (ms)');ax.set_ylabel('Sample ACF')
    axes[0,0].legend(frameon=False,ncol=2,fontsize=7)
    axes[1,0].set_title('Healthy count level');axes[1,0].set_ylabel('Mean count')
    axes[1,1].set_title('Healthy high-occupancy labels');axes[1,1].set_ylabel('Record rule (%)')
    for ax in axes[1]:ax.set_xlabel('Consecutive evaluation eighth');ax.set_xticks(range(1,9))
    save_figure(fig,'cwru_reference_diagnostics')
    summary={'normal_macro_direct':float(rates.loc['Direct count','normal']), 'normal_macro_record':float(rates.loc['Record count','normal']),
             'primary_macro_direct':float(rates.loc['Direct count','primary']), 'primary_macro_record':float(rates.loc['Record count','primary']),
             'relative_reduction_normal_macro':float(1-rates.loc['Record count','normal']/rates.loc['Direct count','normal']),
             'primary_loss_percentage_points':float(100*(rates.loc['Direct count','primary']-rates.loc['Record count','primary'])),
             'all_recordings':56,'primary_fault_recordings':36,'additional_fault_recordings':16,
             'normal_evaluated_blocks':29629,'primary_fault_evaluated_blocks':301367,'additional_fault_evaluated_blocks':126786}
    (OUT/'headline_results.json').write_text(json.dumps(summary,indent=2))
    versions = json.loads((OUT/'software_versions.json').read_text())
    versions['matplotlib'] = matplotlib.__version__
    (OUT/'software_versions.json').write_text(json.dumps(versions,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
