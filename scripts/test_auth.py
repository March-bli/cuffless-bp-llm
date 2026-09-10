import subprocess, re

r1 = subprocess.run(['curl', '-s', '-o', '/dev/null', '-c', '/tmp/test_c.txt',
                     '--connect-timeout', '10',
                     'https://rutgersconnect-my.sharepoint.com/:f:/g/personal/ww329_soe_rutgers_edu/EqalUqc2s_dEqbhgugkUW1MBeNQIUEntgsGM67atFfivbg?e=csitkl'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
print('auth cookie return:', r1.returncode)

r2 = subprocess.run(['curl', '-s', '-b', '/tmp/test_c.txt', '-X', 'POST',
                     '--connect-timeout', '10',
                     '-H', 'Accept: application/json;odata=verbose',
                     '-H', 'Content-Length: 0',
                     'https://rutgersconnect-my.sharepoint.com/personal/ww329_soe_rutgers_edu/_api/contextinfo'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
print('contextinfo return:', r2.returncode, 'len:', len(r2.stdout))
m = re.search(r'FormDigestValue":"(.*?)"', r2.stdout)
print('digest:', bool(m), m.group(1)[:25] if m else 'NONE')
print('stdout head:', r2.stdout[:300])
