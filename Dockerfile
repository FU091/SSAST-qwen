# 1. 決定地基
FROM python:3.8-slim
# 2. 安裝「系統」工具
#    您的 requirements.txt 裡有 librosa (處理音訊)，
#    而 librosa 在讀取某些音訊檔時「需要」ffmpeg。
#    少了這一步，您的 Python 程式在「執行時」(Runtime) 可能會出錯。
RUN apt-get update && apt-get install -y \
 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# 3. 設定工作資料夾
WORKDIR /ssast_hub

#到時候power shell就要先到要讓它變成ssast_hub的來源目錄 cd C:\Users\Lin\Desktop\2_code\ssast_hub
#再建 docker build -t myimage . 


# 4. 把「軟體清單」複製進來
COPY requirements.txt .

# 5. 執行安裝程式
#    (多加了 --upgrade pip，確保 pip 是最新的，避免安裝失敗)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

#RUN pip install \
#    tqdm==4.67.1

# 6. 把要的複製進來
#COPY . /pretrain_dalta_ssl.py
COPY ssast-main/src /ssast_hub/ssast-main/src
#COPY <來源路徑> <目標路徑>


# 7. 再次切路徑後執行主程式（可改成你自己的入口）
# 法1 再次切路徑後執行主程式 WORKDIR /ssast_hub/ssast-main/src接 CMD ["python", "run.py"]
# 法2 直接在原本路徑WORKDIR /ssast_hub 下執行 CMD ["python", "ssast-main/src/run.py"]
CMD ["python", "ssast-main/src/run.py"]