import re

pattern = r'(\w+):\/\/([^/:]+)(:\d*)?([^# ]*)'
# url = 'https://www.example.com:8080/path/to/page.html'
url = 'https://192.168.0.1:8080/path/to/page.html'
match = re.match(pattern, url)
host = ""

if match:
    protocol = match.group(1)
    host = match.group(2)
    port = match.group(3)
    path = match.group(4)

    print("协议:", protocol)
    print("主机:", host)
    print("端口:", port)
    print("路径:", path)
else:
    print("URL 不匹配")


def is_valid_hostname(hostname):
    pattern = r'^[a-zA-Z0-9-]{1,63}(\.[a-zA-Z0-9-]{1,63})*$'
    match = re.match(pattern, hostname)
    return match is not None

# 测试示例
hostnames = ['example.com', 'www.example.com', '123.45.67.89', 'invalid_hostname!']
hostnames.append(host)
for hostname in hostnames:
    if is_valid_hostname(hostname):
        print(f"{hostname} 是一个有效的主机名")
    else:
        print(f"{hostname} 不是一个有效的主机名")