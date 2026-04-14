import re
import json
import math

input_file = "diary_boa.md"
output_file = "diary_boa.jsonl"
MAX_LENGTH = 9999  # 可调，单位为字符
AUTO_WRAP = "n"   # "y" 开启自动按句号换行, "n" 不启用

entries = []
current_entry = []

def remove_links(text):
    """删除文本中的 URL，但保留其他文字"""
    text = re.sub(r"\((https?://[^\s)]+)\)", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    return text.strip()

def split_text_naturally(text, max_len=256):
    """
    按句号尽量拆分文本，每段长度不超过 max_len
    保留原有换行符 \n
    如果句子本身过长或无法拆分，则不拆
    """
    if AUTO_WRAP.lower() != "y":
        return [text]  # 不拆分

    if len(text) <= max_len:
        return [text]

    sentences = re.split(r'(。)', text)
    if len(sentences) <= 1:
        return [text]

    chunks = []
    current = ""
    i = 0
    while i < len(sentences) - 1:
        sentence = sentences[i] + sentences[i + 1]
        sentence = sentence.replace('\n', '\\n')
        if len(current) + len(sentence) <= max_len:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
        i += 2
    if i == len(sentences) - 1:
        last = sentences[-1].replace('\n', '\\n')
        current += last
    if current:
        chunks.append(current.strip())

    return chunks

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].rstrip("\n")
    if line.strip().startswith("#") or line.strip() == "---":
        # 遇到 # 或 --- 时向后查找下一条 # 或 --- 或文件末尾
        if current_entry:
            text = remove_links("\n".join(current_entry))
            chunks = split_text_naturally(text, MAX_LENGTH)
            for chunk in chunks:
                if len(chunk) > MAX_LENGTH:
                    entries.append({
                        "text": f"【**超标 {len(chunk)}，{len(chunk)/MAX_LENGTH:.2f}倍**】\n{chunk}"
                    })
                    # 追加空占位行
                    extra = int(len(chunk) / MAX_LENGTH)
                    for _ in range(extra):
                        entries.append({"text": ""})
                else:
                    entries.append({"text": chunk})
            current_entry = []
        # 跳过连续的 # 或 ---
        while i + 1 < len(lines) and (lines[i + 1].strip().startswith("#") or lines[i + 1].strip() == "---"):
            i += 1
    else:
        current_entry.append(line)
    i += 1

# 文件末尾
if current_entry:
    text = remove_links("\n".join(current_entry))
    chunks = split_text_naturally(text, MAX_LENGTH)
    for chunk in chunks:
        if len(chunk) > MAX_LENGTH:
            entries.append({
                "text": f"【**超标 {len(chunk)}，{len(chunk)/MAX_LENGTH:.2f}倍**】\n{chunk}"
            })
            # 追加空占位行
            extra = int(len(chunk) / MAX_LENGTH)
            for _ in range(extra):
                entries.append({"text": ""})
        else:
            entries.append({"text": chunk})

# 删除无意义的 {"text": ""}
# EVIL CHAT之庆丰逻辑神奇bug
cleaned_entries = []
for idx, e in enumerate(entries):
    if e["text"].strip() == "":
        # 如果前一个是超标文本，就保留这个占位
        if idx > 0 and entries[idx - 1]["text"].startswith("【**超标"):
            cleaned_entries.append(e)
        # 否则跳过
    else:
        cleaned_entries.append(e)

entries = cleaned_entries


# 写 JSONL 文件
with open(output_file, "w", encoding="utf-8") as f:
    # 第一行：标准长度文本
    standard_text = "草" * MAX_LENGTH
    f.write(json.dumps({"text": standard_text}, ensure_ascii=False) + "\n")
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"已生成 JSONL 文件: {output_file}")
