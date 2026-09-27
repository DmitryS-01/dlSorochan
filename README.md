# ДЗ 3 -- Использование Mantis

> Сорочан Дмитрий, ПАДИИ3

## Постановка задачи

Сравниваю Transformer, обученный с нуля, с frozen-признаками Mantis+ на UCI HAR. Дополнительно проверяю, как pretrained representations ведут себя при блочной потере части временного сигнала.

## Вычислительная конфигурация

- device: `mps`;
- accelerator: `Apple Metal Performance Shaders`;
- platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`;
- Python: `3.14.7`;
- PyTorch: `2.14.0`;
- CPU threads: `14`;
- Mantis package: `mantis-tsfm==1.1.0`;
- checkpoint: `paris-noah/MantisPlus`.

## Корпус

UCI HAR: 7352 train / 2947 test, 9 inertial channels, длина окна 128, 6 классов. Official split разделен по людям. Validation внутри train также group-wise по `subject_id`.

## Дополнительный stress-test

У 4 из 9 каналов один непрерывный блок длиной 20%, 35% или 50% заменяется средним по оставшейся наблюдаемой части канала. Модели обучаются на clean train. Stress-test не является missing-aware постановкой: mask модели не получают; он проверяет устойчивость representations к потере локальной динамики.

## Transformer baseline

`Linear(9 -> 96) -> CLS + positional embedding -> 6 Transformer blocks -> classifier`.

Лучший epoch выбирается по clean validation macro F1, после чего baseline заново обучается на всем official train. Лучший validation macro F1 получен на 17-й эпохе (`0.9724`), итоговый official test macro F1 -- `0.9137`.

## Frozen Mantis+

Mantis encoder не обучается. Для каждого из 6 существующих Transformer layers используется одинаковый `StandardScaler + LogisticRegression` на clean features.

## Сравнение слоев

| layer | feature_dim | clean_macro_f1 | robust_mean_f1 | selection_f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4608 | 0.9718 | 0.9602 | 0.9631 |
| 1 | 4608 | 0.9648 | 0.9467 | 0.9513 |
| 2 | 4608 | 0.9716 | 0.9479 | 0.9539 |
| 3 | 4608 | 0.9706 | 0.9498 | 0.9550 |
| 4 | 4608 | 0.9678 | 0.9423 | 0.9487 |
| 5 | 4608 | 0.9693 | 0.9388 | 0.9464 |

Лучший clean layer и лучший layer по среднему clean+stress validation score совпали: `layer 0`.

На clean различия между слоями малы: spread всего `0.0070` macro F1. При усилении corruption spread растет до `0.0411` при 50%. То есть degradation входа делает различия между уровнями representation гораздо заметнее. В этой задаче ранний layer оказывается наиболее устойчивым; более глубокий representation не означает автоматически более transferable features.

## Ablation temporal scales

| scale | feature_dim | clean_macro_f1 | corrupt35_macro_f1 | selection_f1 |
|:---|---:|---:|---:|---:|
| 128 | 4608 | 0.9746 | 0.9617 | 0.9682 |
| 512 | 4608 | 0.9718 | **0.9637** | 0.9678 |
| 128+512 | 9216 | **0.9777** | 0.9592 | **0.9684** |

По формальному среднему clean и 35% corruption побеждает `128+512`, но selection scores отличаются меньше чем на `0.0007`. Multi-scale дает лучший clean result, однако не выигрывает robustness и удваивает размерность признаков. Поэтому его преимущество в этой задаче скорее quality-oriented, чем efficiency-oriented.

## Clean test

| model | accuracy | balanced_accuracy | macro_f1 | selection_time_s | train_time_s | latency_ms_sample |
|:---|---:|---:|---:|---:|---:|---:|
| Transformer | 0.9148 | 0.9152 | 0.9137 | 119.74 | 80.36 | 0.264 |
| Mantis+ frozen (L0, 128+512) | **0.9776** | **0.9772** | **0.9778** | 143.31 | 13.34 | 1.677 |

Frozen Mantis+ улучшает clean macro F1 на **+0.0641**. Особенно заметен выигрыш на трудных классах: `SITTING` растет с `0.8478` до `0.9505` F1, `STANDING` -- с `0.8836` до `0.9598`. Для `LAYING` прирост почти отсутствует, потому что baseline и так близок к насыщению.

## Robustness test

| corruption | Transformer | Mantis+ frozen |
|---:|---:|---:|
| 0% | 0.9137 | **0.9778** |
| 20% | 0.9080 | **0.9705** |
| 35% | 0.8994 | **0.9543** |
| 50% | 0.8848 | **0.9260** |

Mantis остается лучше на всех уровнях corruption, но относительно собственного clean score деградирует сильнее: падение до 50% равно `0.0518`, у Transformer -- `0.0289`. Поэтому stress-test не доказывает missing-aware свойства Mantis. Корректный вывод: pretrained features дают высокий запас абсолютного качества, который постепенно уменьшается по мере потери информации.

## Стоило ли использовать Mantis

На этом корпусе -- **да**. Clean macro F1 растет с `0.9137` до `0.9778`, то есть ошибка `1 - F1` уменьшается примерно на 74%. Encoder при этом не fine-tune'ится на UCI HAR.

По вычислениям trade-off неоднозначный. Полный поиск конфигурации Mantis занимает немного больше времени (`143.3 s` против `119.7 s`), зато финальный downstream fit после выбора конфигурации существенно дешевле (`13.3 s` против `80.4 s`), потому что обучается только linear classifier поверх frozen features. Inference, наоборот, примерно в 6.35 раза медленнее (`1.677` против `0.264 ms/sample`).

Поэтому Mantis особенно логична, когда важны качество, быстрый downstream fit и возможность переиспользовать один encoder для нескольких задач. При жестких edge-ограничениях на latency или при наличии очень сильной специализированной модели выигрыш может не окупиться. Low-data regime отдельно не исследовался, поэтому преимущество при малом числе labels по этому эксперименту не утверждается.

## Итог

1. Frozen Mantis+ существенно превосходит выбранный supervised Transformer baseline на clean UCI HAR.
2. Layer probing становится содержательным под distribution shift: spread между слоями растет вместе с corruption, а `layer 0` остается лучшим.
3. Multi-scale `128+512` дает максимум clean quality, но его преимущество над single-scale минимально и не сопровождается лучшей robustness.
4. Mantis не является missing-aware: при сильной потере информации она деградирует быстрее относительно собственного clean score, хотя абсолютное качество остается выше baseline.
5. Корректный claim ограничен этой постановкой: результат показывает сильный transfer synthetic pretraining -> реальный HAR, но не доказывает универсальное превосходство Mantis над всеми task-specific time-series моделями.

## Где что лежит

| Требование | Где выполнено |
| --- | --- |
| Вычислительная конфигурация | `results/compute_config.json`, `01_Transformer` |
| Корпус | `00_Data_EDA` |
| Baseline | `01_Transformer` |
| Frozen Mantis + classifier | `02_Mantis` |
| Сравнение слоев | `02_Mantis`, `results/02_mantis_layers.csv`, `results/02_mantis_layer_stress.csv` |
| Ablation | `02_Mantis`, `results/02_mantis_ablation.csv` |
| Качество и время | `results/comparison.csv`, графики в `figures/` |
| Robustness | `results/robustness_comparison.csv`, `figures/02_robustness_comparison.png` |
| Вывод | этот файл и финальные markdown-комментарии `02_Mantis` |
