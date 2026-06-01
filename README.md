# s2_cell_aggregate_detection

## 環境構築

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 実行

### 基本（merge画像 + 輪郭オーバーレイのみ出力）
```bash
python detect_aggregations.py input.nd2 output/
```

### よく使うオプションの例
```bash
python detect_aggregations.py input.nd2 output/ \
    --s2-diameter 10.0 \
    --min-cells 3 \
    --min-active-channels 2 \
    --channel-pixel-threshold 10.0 \
    --channel-occupancy-fraction 0.3
```

### デバッグモード（中間マスク・チャンネル別画像をすべて出力）
```bash
python detect_aggregations.py input.nd2 output/ --debug
```

## オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--s2-diameter` | `10.0` | S2細胞の直径（µm）。面積閾値とメディアンフィルタカーネルの計算に使用 |
| `--min-cells` | `3` | アグリゲーションと判定する最小細胞数 |
| `--morph-close-radius` | `5` | モルフォロジー閉処理のディスク半径（px） |
| `--binary-threshold` | `15` | メディアンフィルタ後の固定二値化閾値（0–255）。値を上げると小さな領域が消える |
| `--min-active-channels` | `2` | 採択に必要な最小アクティブチャンネル数（中央スライスの二値化マスクで判定） |
| `--debug` | `False` | 中間画像（チャンネル別MIP・中央スライス・マスク各段階）を追加出力 |

## 出力ファイル

### 常時出力（`output/debug/<ファイル名>/` 以下）
| ファイル名 | 内容 |
|---|---|
| `fieldXX_color_merge.png` | 全チャンネル合成カラー画像（疑似カラーMIP） |
| `fieldXX_overlay_valid.png` | 検出されたアグリゲーションの輪郭（赤） |
| `fieldXX_overlay_rejected.png` | 面積は通過したが棄却された領域の輪郭（青） |
| `<nd2ファイル名>_aggregations.csv` | 検出結果CSV |

### `--debug` 時に追加出力
| ファイル名 | 内容 |
|---|---|
| `fieldXX_ch{i}_mip.png` | チャンネル別MIP（グローバル正規化済み） |
| `fieldXX_ch{i}_center_median.png` | チャンネル別中央Zスライス＋メディアンフィルタ |
| `fieldXX_ch{i}_center_binary.png` | 上記を `binary_threshold` で二値化したもの（チャンネル活性判定に使用） |
| `fieldXX_mip_merged_median.png` | 全チャンネルMIP合成後にメディアンフィルタを適用したグレースケール画像（Otsu入力） |
| `fieldXX_binary_thresh.png` | 固定閾値による二値化マスク |
| `fieldXX_binary_filled.png` | 穴埋め後マスク |
| `fieldXX_binary_closed.png` | モルフォロジー閉処理後の最終マスク |
| `fieldXX_overlay_gray.png` | グレースケール輪郭オーバーレイ |
