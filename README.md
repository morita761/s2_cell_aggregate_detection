# s2_cell_aggregate_detection

## 環境構築

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1. アグリゲーション検出（detect_aggregations.py）

### 基本（merge画像 + 輪郭オーバーレイのみ出力）
```bash
python detect_aggregations.py input.nd2 output/
```

### よく使うオプションの例
```bash
python detect_aggregations.py input.nd2 output/ \
    --s2-diameter 10.0 \
    --min-cells 3 \
    --binary-threshold 50 \
    --min-active-channels 2
```

### デバッグモード（中間マスク・チャンネル別画像をすべて出力）
```bash
python detect_aggregations.py input.nd2 output/ --debug
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--s2-diameter` | `10.0` | S2細胞の直径（µm）。面積閾値とメディアンフィルタカーネルの計算に使用 |
| `--min-cells` | `3` | アグリゲーションと判定する最小細胞数 |
| `--morph-close-radius` | `5` | モルフォロジー閉処理のディスク半径（px） |
| `--binary-threshold` | `50` | メディアンフィルタ後の固定二値化閾値（0–255）。MIPの二値化とチャンネル活性判定の両方に使用 |
| `--min-active-channels` | `2` | 採択に必要な最小アクティブチャンネル数（中央スライスの二値化マスクで判定） |
| `--debug` | `False` | 中間画像（チャンネル別MIP・中央スライス・マスク各段階）を追加出力 |

### 出力ファイル

#### 常時出力（`output/debug/<ファイル名>/` 以下）
| ファイル名 | 内容 |
|---|---|
| `fieldXX_color_merge.png` | 全チャンネル合成カラー画像（疑似カラーMIP） |
| `fieldXX_overlay_valid.png` | 検出されたアグリゲーションの輪郭（赤） |
| `fieldXX_overlay_rejected.png` | 面積は通過したが棄却された領域の輪郭（青） |
| `<nd2ファイル名>_aggregations.csv` | 検出結果CSV |

#### `--debug` 時に追加出力
| ファイル名 | 内容 |
|---|---|
| `fieldXX_ch{i}_mip.png` | チャンネル別MIP（グローバル正規化済み） |
| `fieldXX_ch{i}_center_median.png` | チャンネル別中央Zスライス＋メディアンフィルタ |
| `fieldXX_ch{i}_center_binary.png` | 上記を `binary_threshold` で二値化（チャンネル活性判定に使用） |
| `fieldXX_mip_merged_median.png` | 全チャンネルMIP合成後にメディアンフィルタを適用したグレースケール画像 |
| `fieldXX_binary_thresh.png` | 固定閾値による二値化マスク |
| `fieldXX_binary_filled.png` | 穴埋め後マスク |
| `fieldXX_binary_closed.png` | モルフォロジー閉処理後の最終マスク |
| `fieldXX_overlay_gray.png` | グレースケール輪郭オーバーレイ |

### CSVの列定義
| 列名 | 内容 |
|---|---|
| `field_id` | 視野番号 |
| `aggregation_id` | 視野内のアグリゲーション番号 |
| `area_px` | 面積（ピクセル） |
| `area_um2` | 面積（µm²） |
| `active_channels` | 活性と判定されたチャンネル数 |

---

## 2. 箱ひげ図（tools/boxplot.py）

検出結果のCSVをもとに、条件間の `area_um2` を箱ひげ図＋生データ点で比較します。

### 使い方

```bash
python3 tools/boxplot.py 設定ファイル.txt csvフォルダ1/ csvフォルダ2/ [... 最大6つ]
```

各フォルダを1条件として扱い、フォルダ内の `*_aggregations.csv` を全て読み込みます。

#### 例：3条件を比較してPNGに保存
```bash
python3 tools/boxplot.py tools/config_example.txt \
    output/ctrl/ \
    output/treat_a/ \
    output/treat_b/
```

### 設定ファイル（INI形式）

```ini
[labels]
label1 = Control
label2 = Treatment A
label3 = Treatment B
label4 = Treatment B2

[comparisons]
# 比べたいラベル番号を "N,M" で指定（省略時は隣接グループ間を自動比較）
pair1 = 1,2
pair2 = 1,3
pair3 = 1,4

[plot]
title = S2 Cell Aggregation Size
# output = result.png  ← 省略時は画面表示、指定するとファイル保存
```

- `[comparisons]` を省略すると `1v2, 2v3, 3v4 …` の隣接比較になります
- `pair1 = 1,3` のように飛び越えた比較も可能です
- 有意差バーは範囲の狭いものが下、広いものが上に自動配置されます

テンプレートは `tools/config_example.txt` を参照してください。

### グラフの仕様

| 要素 | 内容 |
|---|---|
| 箱ひげ図 | 白黒・グループごとに異なるハッチパターン |
| ストリッププロット | 半透明の黒点（ジッタ付き）を重ねて表示 |
| n= 表示 | 各グループ上部に生データ数を表示 |
| 有意差 | 隣接グループ間のt検定、`*`〜`*****` / `n.s.` でブラケット表示 |
| 出力 | `[plot] output = ファイル名` を指定するとファイル保存（.png / .pdf / .svg）、省略時は画面表示 |

### エラー処理

以下のケースは処理をスキップしてログに記録します（プログラムは停止しません）。

- フォルダが存在しない
- フォルダ内に `*_aggregations.csv` が見つからない
- CSVに `area_um2` 列がない
- t検定に必要なデータ数が2未満（`n.s.` 表示もスキップ）
