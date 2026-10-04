# Ground-truth verdicts from reading each page image. 2=Correct, 1=Partial, 0=Incorrect
GT = {
'Q01':(0,"3,45%→3,42% is not an increase; 3,45 is SKDU 2017 expectation"),
'Q02':(1,"3,42% & SBT 3,38% are right, but no difference computed; mixes unrelated metrics"),
'Q03':(0,"Keuangan/Real Estate/Jasa Perusahaan = 3,71%; 3,68% is Konstruksi"),
'Q04':(0,"Kredit Korporasi leads contribution (7,41), not konsumsi (1,80)"),
'Q05':(0,"Garbled question; modal kerja fell 10,05→8,35 in 2024, not all rose"),
'Q06':(0,"18,47% is 5-yr avg; growth slowed 28,15%→19,80%"),
'Q07':(0,"Tabungan share 54,4%→54,5%; growth slowed 21,9%→19,9%"),
'Q08':(0,"Deposito share fell to 17,8%; growth slowed 8,31%→6,43%"),
'Q09':(0,"Kembung inflation 32,83% yoy; 6,06% is administered prices"),
'Q10':(0,"Hallucinated — page says no such government policy"),
'Q11':(0,"Garbled question; 2,63% is core inflation"),
'Q12':(0,"PAD Tw III 2020 = Rp2,32 T, not Rp23,32 T"),
'Q13':(0,"PAD realisation −21,03% yoy (target −8,72%), not −5,18%"),
'Q14':(0,"PAD −21,03%, pajak daerah −22,65%; 12,68% is total revenue, 6,12% is 2019"),
'Q15':(1,"Values 5,63→5,93 right but periods are Tw I→Tw II 2015, not 2014→2015"),
'Q16':(0,"8,72% is cigarette excise; base effect refers to tarif listrik"),
'Q17':(0,"Kretek filter inflation 13,52%, andil 0,30"),
'Q18':(2,"Passenger departures +19,88% yoy Tw III 2016 (question garbled)"),
'Q19':(0,"International cargo +0,87%; 8,26% is sea passengers"),
'Q20':(1,"Qualitatively matches text but question asks 'berapa' — no figure"),
'Q21':(0,"4,82% is TPT (unemployment), not agriculture; prior was 3,43%"),
'Q22':(2,"TPAK 67,83%→69,48% correct"),
'Q23':(0,"Performance declined; 88,54% is Tw III LDR"),
'Q24':(1,"Rp384,01 T nominal right, but growth is −0,95% yoy"),
'Q25':(0,"70,62% is DKI share of Java DPK"),
'Q26':(2,"3,71% correct"),
'Q27':(2,"Listrik, Gas dan Air Bersih correct"),
'Q28':(1,"Text says credit 'perlu terus ditingkatkan' — a need, not a fact"),
'Q29':(1,"9,69% vs 10,39% & drivers correct, but doesn't address macroprudential response"),
'Q30':(2,"0,99 correct"),
'Q31':(2,"34,0% correct"),
'Q32':(2,"Rp7,16 T correct"),
'Q33':(2,"9,0% correct"),
'Q34':(0,"Highest yoy is Tembang 67,95%; highest sum-yoy Daging Ayam Ras"),
'Q35':(1,"Bawang Merah is right commodity but rate (−33,89%) not given"),
'Q36':(2,"Rokok inflation present — correct"),
'Q37':(2,"Revenue contracted 12,68% yoy — correct"),
'Q38':(0,"Serapan 72,24% vs 77,42% — lower, not comparable; 77,24 misread"),
'Q39':(1,"Direction right; −8,72% is target, realisation −21,03%"),
'Q40':(2,"Tarif listrik correct"),
'Q41':(2,"~70 juta stagnant, cukai +8,72% correct"),
'Q42':(2,"Telepon seluler correct (awkward wording)"),
'Q43':(0,"7,64% is transport sector growth, not tourist visits"),
'Q44':(0,"Unit/number nonsense — chart is in ribu orang (~2.800)"),
'Q45':(2,"Sea passenger growth 8,26% correct"),
'Q46':(2,"69,48% correct"),
'Q47':(1,"Values right; increase is Feb 2016→Feb 2017, not 'on Feb 2016'"),
'Q48':(2,"Menurun correct"),
'Q49':(0,"Largest share is DKI Jakarta (70,79%), not Jawa Barat"),
'Q50':(2,"Menurun correct"),
}
import json, numpy as np, pandas as pd, itertools, krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from sklearn.metrics import cohen_kappa_score

items=json.load(open('sample50.json'))
ids=[it['id'] for it in items]; gt=np.array([GT[i][0] for i in ids]); model=np.array([it['model'] for it in items])
LABELS={0:'Incorrect',1:'Partial',2:'Correct'}
# Annotator profiles: base accuracy + bias (+1 lenient / -1 strict / 0 neutral) for errors
ANN=[('A1 (expert)',0.92,0),('A2',0.88,0),('A3',0.85,-1),('A4',0.80,0),('A5 (lenient)',0.78,+1)]
PARTIAL_PENALTY=0.15   # 'Partial' items are harder to judge
def simulate(rng):
    R=np.zeros((len(gt),len(ANN)),int)
    for a,(n,acc,bias) in enumerate(ANN):
        for i,g in enumerate(gt):
            p=acc-(PARTIAL_PENALTY if g==1 else 0)
            if rng.random()<p: R[i,a]=g; continue
            opts=[l for l in (0,1,2) if l!=g]
            w=np.array([1.0 if abs(l-g)==1 else 0.3 for l in opts])   # adjacent errors more likely
            if bias: w*=np.array([2.0 if np.sign(l-g)==bias else 1.0 for l in opts])
            R[i,a]=rng.choice(opts,p=w/w.sum())
    return R
def metrics(R,g):
    t,_=aggregate_raters(R,n_cat=3)
    fk=fleiss_kappa(t) if len(set(R.ravel()))>1 else np.nan
    ka=krippendorff.alpha(reliability_data=R.T,level_of_measurement='ordinal')
    pc=np.mean([cohen_kappa_score(R[:,a],R[:,b]) for a,b in itertools.combinations(range(R.shape[1]),2)])
    pw=np.mean([cohen_kappa_score(R[:,a],R[:,b],weights='quadratic') for a,b in itertools.combinations(range(R.shape[1]),2)])
    maj=np.array([np.bincount(r,minlength=3).argmax() for r in R])
    return dict(fleiss=fk,kripp_ord=ka,pair_cohen=pc,pair_wquad=pw,maj_acc=(maj==g).mean(),
                maj_vs_gt_kappa=cohen_kappa_score(maj,g),pct_unanimous=np.mean([len(set(r))==1 for r in R]))
# --- one reported run
rng=np.random.default_rng(2026); R=simulate(rng)
run={'All (n=50)':metrics(R,gt)}
for m in ['InternVL','Qwen2.5-VL']:
    k=model==m; run[f'{m} (n={k.sum()})']=metrics(R[k],gt[k])
per_ann=[dict(annotator=n,acc_vs_gt=(R[:,a]==gt).mean(),cohen_vs_gt=cohen_kappa_score(R[:,a],gt),
              wkappa_vs_gt=cohen_kappa_score(R[:,a],gt,weights='quadratic')) for a,(n,_,_) in enumerate(ANN)]
pair=pd.DataFrame(np.eye(5),index=[a[0] for a in ANN],columns=[a[0] for a in ANN])
for a,b in itertools.combinations(range(5),2):
    pair.iloc[a,b]=pair.iloc[b,a]=cohen_kappa_score(R[:,a],R[:,b])
# --- Monte Carlo
mc=[]; rng=np.random.default_rng(0)
for _ in range(1000):
    Rm=simulate(rng); d={'All':metrics(Rm,gt)['fleiss']}
    for m in ['InternVL','Qwen2.5-VL']:
        k=model==m; d[m]=metrics(Rm[k],gt[k])['fleiss']
    d['kripp_all']=metrics(Rm,gt)['kripp_ord']; mc.append(d)
mc=pd.DataFrame(mc)
mcs=pd.DataFrame({c:[mc[c].mean(),mc[c].quantile(.025),mc[c].quantile(.975)] for c in mc},index=['mean','p2.5','p97.5']).T
# --- outputs
pd.set_option('display.width',200); pd.set_option('display.precision',3)
print(pd.DataFrame(run).T); print(pd.DataFrame(per_ann)); print(pair.round(3)); print(mcs)
print('GT dist',{m:dict(zip(*np.unique(gt[model==m],return_counts=True))) for m in ['InternVL','Qwen2.5-VL']})
df=pd.DataFrame([dict(id=it['id'],model=it['model'],image=it['file'],category=it['category'],query=it['query'],answer=it['answer'],
    gt_label=LABELS[GT[it['id']][0]],gt_score=GT[it['id']][0],gt_note=GT[it['id']][1],
    **{ANN[a][0]:LABELS[R[i,a]] for a in range(5)},
    majority=LABELS[np.bincount(R[i],minlength=3).argmax()],n_agree_with_gt=int((R[i]==gt[i]).sum())) for i,it in enumerate(items)])
with pd.ExcelWriter('/mnt/user-data/outputs/kappa_simulation_50qa.xlsx') as w:
    summ=pd.DataFrame(run).T; summ.index.name='subset'; summ.to_excel(w,sheet_name='Summary')
    mcs.to_excel(w,sheet_name='MonteCarlo_1000runs')
    df.to_excel(w,sheet_name='Annotations',index=False)
    pd.DataFrame(per_ann).to_excel(w,sheet_name='Annotator_vs_GT',index=False)
    pair.to_excel(w,sheet_name='Pairwise_Cohen')
    pd.DataFrame([dict(annotator=n,base_accuracy=a,error_bias={0:'neutral',1:'lenient',-1:'strict'}[b]) for n,a,b in ANN]+
                 [dict(annotator='note',base_accuracy=f'-{PARTIAL_PENALTY} on Partial items',error_bias='adjacent-label errors 3.3x likelier than 0↔2')]).to_excel(w,sheet_name='Sim_Settings',index=False)
json.dump(dict(run=run,per_ann=per_ann,mc=mcs.to_dict()),open('results.json','w'),default=float,indent=1)
