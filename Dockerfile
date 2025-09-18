# 使用官方Python镜像作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 将当前目录中的所有文件复制到容器的工作目录
COPY . /app


# 安装依赖包，使用--no-cache-dir，避免保存不必要文件
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# 暴露端口
EXPOSE 6003

# 启动 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6003"]
