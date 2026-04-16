### Pitch Histograms

Install...

`pip install -r requirements.txt`

Extract pitch track

```
python main.py extract \
    "AlaruluGuriyaga-Sankarabharanam-NRK-TA.mp3" \
    "PullumSilambina-Sankarabharanam-MLV-Andal.wav" \
    "SakhiyeInda-Sankarabharanam-MLV-KNDP.wav" \
    "SamiNinneKori- Shankarabharanam-PRV-VK.mp3" \
    "Shankaracharyam-Sankarabharanam-TDTB-SD-onlysong.wav" \
    "Chamela-Sankarabharanam-RV-ST-onlysong.wav" \
    "SriShanmukha-Sankarabharanam-MSS-MSR-onlysong.wav" \
    "DeviJagajjanani-Sankarabharanam-KOK-ST.wav" \
    "Thillana-Sankarabharanam-TMK-Poochi-onlysong.wav" \
    "ThookiyaThiruvadi-Sankarabharanam-NM-SB-onlysong.wav" \
    "Gurucharanam-Sankarabharanam-TMS-WVB-onlysong.wav" \
    "JananiNatajana-VS-MDR-onlysong.wav" \
    "Mahalakshmi-Sankarabharanam-MS-PS.wav" \
    "nannubrOcuTakevarennAru-Sankarabharanam-SS-MV.wav" \
    "Muthukumarayyane-Sankarabharanam-SR-RS-onlysong.wav" \
    "Papa-PMI-trimmed-Swararagasudha.mp3"
```

Compute histogram automatically extract tonic:

```
python main.py histogram \
    "AlaruluGuriyaga-Sankarabharanam-NRK-TA.tsv" \
    "Bhaktaparayana-Sankarabharanam-AV-ST-onlysong.tsv" \
    "Muthukumarayyane-Sankarabharanam-SR-RS-onlysong.tsv" \
    "PullumSilambina-Sankarabharanam-MLV-Andal.tsv" \
    "SakhiyeInda-Sankarabharanam-MLV-KNDP.tsv" \
    "SamiNinneKori- Shankarabharanam-PRV-VK.tsv" \
    "Shankaracharyam-Sankarabharanam-TDTB-SD-onlysong.tsv" 
```

To parameterise

`python main.py parameterise combined.png all_shankabharanam.dat`
