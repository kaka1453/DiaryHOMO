import re
import json

input_file = "diary_boa.md"
output_file = "diary_boa.jsonl"

# 匹配日记标题，如：
# ## 2024/9/24
# ## 2025/1/13 周一
pattern = re.compile(
    r"^##\s*\d{4}/\d{1,2}/\d{1,2}(?:\s*周[一二三四五六日天])?",
    re.MULTILINE
)

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

matches = list(pattern.finditer(content))
entries = []

for i, match in enumerate(matches):
    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

    entry_text = content[start:end]

    # ===== 1️⃣ 去掉段首的日期标题 =====
    entry_text = pattern.sub("", entry_text, count=1)

    # ===== 2️⃣ 只去掉段首的换行 =====
    entry_text = re.sub(r"^\n+", "", entry_text)

    # ===== 3️⃣ 只去掉段尾的换行 =====
    entry_text = re.sub(r"\n+$", "", entry_text)

    # 防止空段
    if entry_text.strip():
        entries.append({"text": entry_text})

with open(output_file, "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 已生成 JSONL 文件: {output_file}")
