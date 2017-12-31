# Rugby Classication
## Requirements
- Python 3.6.1
- Chainer 3.2.0
- Cupy 2.2.0

## セットアップ
```
git clone https://github.com/yasunorikudo/ichige.git
cd ichige
pip install -r requirements.txt
```
## 学習
- 学習用データ[2,3,4]・評価用データ[1]
  ```
  python scripts/train.py -t 1 2 3 -v 4
  ```
- 学習用データ[1,3,4]・評価用データ[2]
  ```
  python scripts/train.py -t 1 2 3 -v 4
  ```
- 学習用データ[1,2,4]・評価用データ[3]
  ```
  python scripts/train.py -t 1 2 3 -v 4
  ```
- 学習用データ[1,2,3]・評価用データ[4]
  ```
  python scripts/train.py -t 1 2 3 -v 4
  ```
- その他オプション・ヘルプ
  ```
  python scripts/train.py --help
  ```
