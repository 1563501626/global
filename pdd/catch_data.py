import json
import pandas as pd
import datetime
import re

with open('goods.txt', 'r', encoding='utf8') as f:
    data = f.read()

maps = {
    'goods_id': '商品id',
    'goods_name': '商品名称',
    'hd_thumb_url': '封面图片连接',
    'hd_url': '封面大图片连接',
    'link_url': '连接地址',
    'market_price': '市场价',
    'normal_price': '普通价',
    'price': '封面价',
    'sales': '销售额',
    'sales_tip': '已拼件数',
    'short_name': '商品名称',
    'saletime': '上货时间（应该是吧，图片链接中提取的）',
    'datenow': '数据更新时间'
}


'''list(map(lambda x: maps[x], ['goods_id', 'goods_name', 'hd_thumb_url', 'hd_url', 'link_url', 'market_price',
                           'normal_price', 'price', 'sales', 'sales_tip', 'short_name', 'saletime', 'datenow']))'''


content_li = data.split('\n')
df = pd.DataFrame(columns=['goods_id', 'goods_name', 'hd_thumb_url', 'hd_url', 'link_url', 'market_price',
                           'normal_price', 'price', 'sales', 'sales_tip', 'short_name', 'saletime', 'datenow'])
data = []
for i in content_li:
    i = json.loads(i)
    goods_li = i['items']
    for j in goods_li:
        goods_id = j['goods_id']
        goods_name = j['goods_name']
        hd_thumb_url = j['hd_thumb_url']
        hd_url = j['hd_url']
        link_url = j['link_url']
        market_price = j['market_price']
        normal_price = j['normal_price']
        price = j['price']  # 封面显示价格
        sales = j['sales']  # 销售额
        sales_tip = j['sales_tip']  # 已拼件数
        short_name = j['short_name']
        saletime = re.search(r'\d{4}-\d+-\d+', hd_thumb_url).group()
        datenow = datetime.datetime.now()
        data.append([goods_id, goods_name, hd_thumb_url, hd_url, link_url, market_price,
                           normal_price, price, sales, sales_tip, short_name, saletime, datenow])

df = pd.DataFrame(data, columns=list(map(lambda x: maps[x], ['goods_id', 'goods_name', 'hd_thumb_url', 'hd_url', 'link_url', 'market_price',
                           'normal_price', 'price', 'sales', 'sales_tip', 'short_name', 'saletime', 'datenow'])))

df.to_csv('./goods.csv', index=False, encoding='gbk')