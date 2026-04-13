import re
import tiktoken

input_file = "yuanzhikong.md"
output_file = "yuanzhikong_checked1.md"
MAX_TOKENS = 512  # 以 token 计数为阈值

# 初始化 BPE tokenizer
# 这里用 cl100k_base，可根据你的模型替换
encoder = tiktoken.get_encoding("cl100k_base")

# 匹配日记标题的正则
pattern = re.compile(r"^##\s*\d{4}/\d{1,2}/\d{1,2}(?:\s*周[一二三四五六日天])?", re.MULTILINE)

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 找到所有标题位置
matches = list(pattern.finditer(content))
new_content = ""
for i, match in enumerate(matches):
    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
    entry_text = content[start:end].strip()

    # 提取正文部分（去掉标题本身）
    lines = entry_text.splitlines()
    if len(lines) > 1:
        body = "\n".join(lines[1:]).strip()
    else:
        body = ""

    # 使用 BPE tokenizer 计算 token 数
    token_count = len(encoder.encode(body))

    # 检测是否超标
    if token_count > MAX_TOKENS:
        header = lines[0]
        warning = f"【**超标 {token_count} token，{token_count / MAX_TOKENS:.2f}倍**】"
        entry_text = f"{header}\n{warning}\n" + "\n".join(lines[1:])

    new_content += entry_text + "\n\n"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"✅ 检查完成，已输出到 {output_file}")
