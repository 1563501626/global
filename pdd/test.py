import requests
import execjs
import re
from lxml import etree
import os

selector = etree.HTML

s = requests.session()
for i in range(1, 10):
    page = i
    flip = (page-1)*20

    if page == 1:
        url = 'http://mobile.yangkeduo.com/search_result.html?search_key=%E5%A5%B3%E8%A3%852019%E6%96%B0%E6%AC%BE%E6%BD%AE%E5%88%9D%E7%A7%8B%E5%AD%A6%E7%94%9F%E5%A4%96%E5%A5%97&search_src=history&search_met=history_sort&search_met_track=history&refer_search_met_pos=0&refer_page_name=search&refer_page_id=10031_1565068587591_krWpAI94kc&refer_page_sn=10031'
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Cookie': 'api_uid=rBUoKV17vXuA+QxhVfB5Ag==; ua=Mozilla%2F5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F76.0.3809.100%20Safari%2F537.36; _nano_fp=Xpdjl0XbX0dxn5TxnC_buu9ZzWCx40O3JVzSVGz9; webp=1; pdd_user_id=2594367865969; pdd_user_uin=FLABGUDKFYAJZELBHF6PK4C4ZQ_GEXDA; PDDAccessToken=DU65WZEZ4WYWQ2QXHDXU5VPBKPL752SFKOSZJ47WL6MNCVSEBNEA1007115; goods_detail_mall=goods_detail_mall_J5jOpp; msec=1800000',
            'Host': 'mobile.yangkeduo.com',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36',
        }
        res = s.get(url, headers=headers)
        ret = selector(res.content.decode())
        print(url)
        content = ret.xpath("//script[@id='__NEXT_DATA__']/text()")[0]
        with open(r'goods.txt', 'a', encoding='utf8') as f:
            f.write(content + '\n')
    else:
        headers = {
        'AccessToken':'65RSQSVXE744XNWUNNJRHLXV7VLCZC4GECN2MN3YC7CTPYNBGYKA1007115',
        'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8',
        'Referer':'http://mobile.yangkeduo.com/search_result.html?search_key=%E5%A5%B3%E8%A3%852019%E6%96%B0%E6%AC%BE%E6%BD%AE%E5%88%9D%E7%A7%8B%E5%AD%A6%E7%94%9F%E5%A4%96%E5%A5%97&search_src=history&search_met=history_sort&search_met_track=history&refer_search_met_pos=0&refer_page_name=search&refer_page_id=10031_1565068587591_krWpAI94kc&refer_page_sn=10031',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36',
        'VerifyAuthToken':'9K504oAgTBgk5nB_-O8L4g',
        }
        xx = input("input:")
        x = xx
        url = 'http://mobile.yangkeduo.com/proxy/api/search?source=search&search_met=history&track_data=refer_page_id,10031_1565068587591_krWpAI94kc;refer_search_met_pos,0&list_id=83s2UlZeYA&sort=default&filter=&q=%E5%A5%B3%E8%A3%852019%E6%96%B0%E6%AC%BE%E6%BD%AE%E5%88%9D%E7%A7%8B%E5%AD%A6%E7%94%9F%E5%A4%96%E5%A5%97&page=2&size=50&flip=20;4;0;0;e21f8621-4824-44f8-a0ca-a3024fdc8d2a&anti_content=' + x
        print(url)
        res = requests.get(url, headers=headers)
        resp = res.content.decode()
        with open(r'goods.txt', 'a', encoding='utf8') as f:
            f.write(resp+'\n')
        print()
        '{"server_time":1568702144,"verify_auth_token":"B_4eNVu8Fab2l45ovu_YlA","scene_id":5,"error_code":54001}'
        '{"server_time":1568702392,"verify_auth_token":"R3JP79L9lvm_bWevV43-hg","scene_id":5,"error_code":54001}'

#
# from pyppeteer import launch
# import asyncio
# import time
#
#
# f = open(r'goods.txt', 'w', encoding='utf8')
#
#
# async def main():
#     url = 'http://yangkeduo.com/search_result.html?search_key=%E5%A5%B3%E8%A3%852019%E6%96%B0%E6%AC%BE%E6%BD%AE%E5%88%9D%E7%A7%8B%E5%AD%A6%E7%94%9F%E5%A4%96%E5%A5%97&search_src=history&search_met=history_sort&search_met_track=history&refer_search_met_pos=0&refer_page_name=login&refer_page_id=10169_1568696256076_p3oAVecJOg&refer_page_sn=10169'
#     browser = await launch({'headless': False,  # 是否无头
#          'ignorehttpserrrors': True,  # 是否忽略https错误
#          'args': ['--disable-infobars',  # 关闭 受控制提示
#                   '--proxy-server=106.12.199.193:8089'
#                   ]  # 代理
#          })
#     page = await browser.newPage()
#
#     coos = [{'name':'api_uid', 'value':'Cg+zQl1/EiKQ9wA9uwGTAg=='},
#     {'name':'_nano_fp', 'value':'Xpdjl0Eol0CalpdJXo_h~UkZgPmzMLQFcNCqSaJs',},
#     {'name':'ua', 'value':'Mozilla%2F5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F77.0.3865.75%20Safari%2F537.36',},
#     {'name':'JSESSIONID', 'value':'6062F0F269C9AFA3689FA31DAE9730F4',},
#     {'name':'webp', 'value':'1',},
#     {'name':'goods_detail_mall', 'value':'goods_detail_mall_xBpD1p',},
#     {'name':'msec', 'value':'1800000',},
#     {'name':'pdd_user_id', 'value':'2594367865969'},
#     {'name':'pdd_user_uin', 'value':'FLABGUDKFYAJZELBHF6PK4C4ZQ_GEXDA'},
#     {'name':'PDDAccessToken', 'value':'ICSMF4ZDCP3BKH2UHLBQJQRJJMHUY6TPR4JN36JI4OGPTPKCLIIQ1007115'},]
#         # 'api_uid':'Cg+zQl1/EiKQ9wA9uwGTAg==',
#         # '_nano_fp':'Xpdjl0ExXqUyX5TbXT_fUwFLtJK9WmJ3vgy~5a2j',
#         # 'ua':'Mozilla%2F5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F77.0.3865.75%20Safari%2F537.36',
#         # 'webp':'1',
#         # 'JSESSIONID':'D0AF553909A996D5864082BD902D37DD',
#         # 'pdd_user_id':'2594367865969',
#         # 'pdd_user_uin':'FLABGUDKFYAJZELBHF6PK4C4ZQ_GEXDA',
#         # 'PDDAccessToken':'HWPOKRAQW7YAYB7MP3GQX73NQXXDSKVLZJHXXVRUNFSMC7HAEHZA1007115',
#
#     # await page.setViewport({'width': width, 'height': height})
#     await page.goto(url, {'timeout': 1000 * 60})
#     for coo in coos:
#         await page.setCookie(coo)
#     await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
#                             ' Chrome/74.0.3729.169 Safari/537.36')
#     await page.evaluate(
#         '''() =>{ Object.defineProperties(navigator,{ webdriver:{ get: () => false } }) }''')
#
#     await page.waitFor(3000)
#     time.sleep(60)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     # login = await page.xpath("//div[@class='phone-login']")
#     # await login[0].click()
#     # page.mouse
#     # time.sleep(2)
#     # inputs = await page.xpath("//input[@id='user-mobile']")
#     # await inputs[0].type("13164647453")
#     # pwd = await page.xpath("//input[@id='input-code']")
#     # await pwd[0].type("846142")
#     # time.sleep(20)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
#     time.sleep(2)
#     a = await page.xpath("//div[@class='hebsNcAM']")
#     yy = await (await a[0].getProperty("textContent")).jsonValue()
#     f.write(yy)
#     print(yy)
#     await browser.close()
#     f.close()
# asyncio.get_event_loop().run_until_complete(main())

































