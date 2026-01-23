param(
    [string]$FilePath = "Soundscape_Checkdata_20250425_T.csv"
)

# 讀取前 10KB
$bytes = Get-Content $FilePath -Encoding Byte -TotalCount 10000
$tmpFile = "$env:TEMP\sample_bytes.bin"
[System.IO.File]::WriteAllBytes($tmpFile, $bytes)

# 呼叫 Python 腳本檢測編碼
python -c @"
import chardet
raw = open(r"$tmpFile","rb").read()
result = chardet.detect(raw)
print("偵測結果:", result)
"@