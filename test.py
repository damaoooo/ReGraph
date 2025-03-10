import requests

url = "https://sstxc.com/mobile/dealings/setaddress.html"

headers = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "dnt": "1",
    "origin": "https://sstxc.com",
    "priority": "u=1, i",
    "referer": "https://sstxc.com/mobile/dealings/setaddress.html",
    "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "x-csrf-token": "7f74a154097551ffc7876fcc698436b5",
    "x-requested-with": "XMLHttpRequest"
}

cookies = {
    "lang": "en-us",
    "PHPSESSID": "3e87ea0fbff9ce8fded3d4d8fbed3063"
}

data = {
    "product_id": "6",
    "withdraw_erc_address": "0xc371295289B009A636a4DC620476eAE55DA66633",
    "withdraw_trc_address": "TNzFUurtU3j8WjTqW5ZyJzDLaXx5AmXsPn",
    "paypwd": "114514"
}

response = requests.post(url, headers=headers, cookies=cookies, data=data)

print(response.text)