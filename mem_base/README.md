## User based, Item based Collaborative Filtering
- 내적으로 유사도 구함
- 0, 1 로만 구성되어 있어 단위 벡터 느낌이라 내적으로 유사도 체크 가능함.
- k는 이웃 개수, 이웃들의 벡터를 단순 평균함.
- 이후 자기가 봤던 것을 제외하고 상위 10개 추천함.

```
# validation
python mbcf.py --k 200

# submission
python mbcf.py --k 200 --inference 1

# item-based
python mbcf.py --k 200 --item_base 1
```