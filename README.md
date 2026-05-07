# s2_cell_aggregate_detection

## 環境構築

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 実行
```bash
python3 detect_aggregations.py 20260501_w_1h_6daysaftertrafec_FSmCherryvsLBGFP001.nd2 output/ \
    --threshold 50 \
    --area-threshold 400 \
    --gaussian-ksize 25 \
    --median-ksize 25


python3 detect_aggregations.py 20260501_w_1h30m_6daysaftertrafec_FSmCherryvsLBGFP003.nd2 output/ \
    --threshold 50 \
    --area-threshold 400 \
    --gaussian-ksize 25 \
    --median-ksize 25    
```