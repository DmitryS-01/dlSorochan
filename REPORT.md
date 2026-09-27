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

Epoch выбирается по clean validation macro F1, после чего baseline заново обучается на всем official train.

## Frozen Mantis+

Mantis encoder не обучается. Для каждого из 6 существующих Transformer layers используется одинаковый `StandardScaler + LogisticRegression` на clean features.

## Сравнение слоев

|   layer |   feature_dim |   extract_fit_val_s |   classifier_fit_s |   clean_macro_f1 |   robust_mean_f1 |   selection_f1 |
|--------:|--------------:|--------------------:|-------------------:|-----------------:|-----------------:|---------------:|
|  0.0000 |     4608.0000 |             12.9164 |             0.5632 |           0.9718 |           0.9602 |         0.9631 |
|  1.0000 |     4608.0000 |             16.2134 |             0.3849 |           0.9648 |           0.9467 |         0.9513 |
|  2.0000 |     4608.0000 |             20.4885 |             0.6457 |           0.9716 |           0.9479 |         0.9539 |
|  3.0000 |     4608.0000 |             24.4471 |             0.4839 |           0.9706 |           0.9498 |         0.9550 |
|  4.0000 |     4608.0000 |             26.0253 |             0.5110 |           0.9678 |           0.9423 |         0.9487 |
|  5.0000 |     4608.0000 |             32.4310 |             0.5050 |           0.9693 |           0.9388 |         0.9464 |

Лучший clean layer: `0`. Лучший layer по среднему clean+stress validation score: `0`.

На clean различия между слоями малы, но при усилении corruption разброс увеличивается. В этой задаче layer 0 остается наиболее устойчивым; более глубокий слой не означает автоматически более transferable representation.

## Ablation temporal scales

| scale   |   feature_dim |   classifier_fit_s |   clean_macro_f1 |   corrupt35_macro_f1 |   selection_f1 |
|:--------|--------------:|-------------------:|-----------------:|---------------------:|---------------:|
| 128     |          4608 |             0.5653 |           0.9746 |               0.9617 |         0.9682 |
| 512     |          4608 |             0.5786 |           0.9718 |               0.9637 |         0.9678 |
| 128+512 |          9216 |             0.8768 |           0.9777 |               0.9592 |         0.9684 |

Лучший scale по формальному среднему clean и 35% corrupted validation: `128+512`. При этом selection scores всех трех вариантов очень близки, а multi-scale удваивает размерность features, поэтому его преимущество скорее quality-oriented, чем efficiency-oriented.

## Clean test

| model                        |   accuracy |   balanced_accuracy |   macro_f1 |   selection_time_s |   train_time_s |   latency_ms_sample |
|:-----------------------------|-----------:|--------------------:|-----------:|-------------------:|---------------:|--------------------:|
| Transformer                  |     0.9148 |              0.9152 |     0.9137 |           114.0085 |        78.3334 |              0.2025 |
| Mantis+ frozen (L0, 128+512) |     0.9776 |              0.9772 |     0.9778 |           146.3879 |        13.6250 |              1.6975 |

## Robustness test

|   corruption |   Mantis+ frozen |   Transformer |
|-------------:|-----------------:|--------------:|
|       0.0000 |           0.9778 |        0.9137 |
|       0.2000 |           0.9705 |        0.9080 |
|       0.3500 |           0.9543 |        0.8994 |
|       0.5000 |           0.9260 |        0.8848 |

## Вывод

Frozen Mantis+ улучшил clean macro F1 на +0.0641.

На clean test Mantis дает `0.9778` macro F1 против `0.9137` у Transformer. Для данного корпуса transfer действительно оправдан: frozen encoder без downstream fine-tuning дает заметно более сильное межсубъектное representation.

При 50% block corruption Transformer получает `0.8848`, Mantis -- `0.9260`. Mantis остается лучше в абсолютном качестве, но ее падение от clean (`0.0518`) больше, чем у Transformer (`0.0289`). Поэтому stress-test не показывает missing-aware свойства Mantis; он показывает высокий запас качества pretrained features, который постепенно уменьшается по мере потери информации.

Layer probing оказался содержательнее именно под corruption: clean scores близки, тогда как при сильной деградации входа разброс между слоями заметно возрастает. В этом эксперименте лучший ранний layer одновременно оптимален и на clean, и по robustness-критерию.

Temporal-scale ablation не дает однозначного robustness-победителя. Multi-scale `128+512` максимизирует clean quality и формально выигрывает средний selection score, но его преимущество над single-scale вариантами минимально при удвоенной размерности features.

По вычислениям Mantis дороже на inference, но дешевле на финальном downstream fit после выбора конфигурации: encoder frozen, поэтому обучается только linear classifier. Следовательно, подход особенно удобен, когда важны качество и переиспользование одного pretrained encoder для нескольких downstream экспериментов. Для жесткого edge-inference или при наличии очень сильной task-specific модели trade-off может оказаться менее выгодным.

Главное ограничение: сравнение проведено на одном корпусе и с одним Transformer baseline. Поэтому корректный claim -- Mantis существенно лучше **в этой постановке**, а не универсально лучше любых специализированных моделей временных рядов. Low-data regime отдельно не исследовался, поэтому утверждать преимущество Mantis при малом числе labels по этому эксперименту нельзя.

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
